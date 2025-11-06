"""Evaluation script for DINOv2/DINOv3 trained models on generated image subsets.

This script:
1. Loads a trained DINOv2/DINOv3 checkpoint (auto-detects model architecture)
2. Reads generated images from folder structure (organized by taxonomy)
3. Maps folder names (with category IDs) to correct labels using training JSON
4. Computes classification accuracy (top-1, top-5)

Usage:
    # Simple usage (auto-detects model and img_size from checkpoint):
    python eval_dinov3_subset.py \
        --checkpoint /path/to/checkpoint_best.pth \
        --train_json /path/to/train.json \
        --sample_dir /fs/scratch/.../generated_images \
        --batch_size 32
    
    # Or override model/img_size if checkpoint doesn't have this info:
    python eval_dinov3_subset.py \
        --checkpoint /path/to/checkpoint_best.pth \
        --train_json /path/to/train.json \
        --sample_dir /fs/scratch/.../generated_images \
        --model_name vit_base_patch16_dinov2.lvd142m \
        --img_size 224 \
        --batch_size 32 \
        --output_json results.json

Why train_json is needed:
    The generated images are in folders like "03111_Animalia_Chordata_..._Accipiter_badius".
    We need to map category ID 03111 -> correct label index (0-9999) that the model expects.
    The train_json provides this mapping via its 'categories' field.

Folder structure expected:
    sample_dir/
        03111_Animalia_Chordata_Aves_..._Accipiter_badius/
            03111_..._sample00.png
            03111_..._sample01.png
        03117_Animalia_Chordata_Aves_..._Aegypius_monachus/
            ...
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import timm


class SubsetImageDataset(Dataset):
    """Dataset that loads images from folder structure with category ID prefixes.
    
    Folders are named: {category_id}_{taxonomy_path}/
    Images inside: {category_id}_{taxonomy_path}_sample{N}.png
    """
    
    def __init__(
        self, 
        sample_dir: str, 
        cat_id_to_label: Dict[int, int],
        transform=None
    ):
        self.sample_dir = Path(sample_dir)
        self.transform = transform
        self.samples = []
        
        # Iterate through folders
        for folder in sorted(self.sample_dir.iterdir()):
            if not folder.is_dir():
                continue
            
            # Extract category ID from folder name (e.g., "03111_Animalia_..." -> 3111)
            folder_name = folder.name
            try:
                cat_id = int(folder_name.split('_')[0])
            except (ValueError, IndexError):
                print(f"Warning: Could not parse category ID from folder: {folder_name}")
                continue
            
            # Check if this category exists in our training set
            if cat_id not in cat_id_to_label:
                print(f"Warning: Category {cat_id} not found in training JSON, skipping folder {folder_name}")
                continue
            
            label = cat_id_to_label[cat_id]
            
            # Add all images in this folder
            for img_path in sorted(folder.glob('*.png')) + sorted(folder.glob('*.jpg')):
                self.samples.append((img_path, label, cat_id))
        
        print(f"Loaded {len(self.samples)} images from {len(set(s[2] for s in self.samples))} categories")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, cat_id = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, cat_id


class DINOv3Classifier(nn.Module):
    """Wrapper: timm DINOv2/DINOv3 backbone + LayerNorm + linear classification head.
    
    Compatible with both DINOv2 and DINOv3 models from timm.
    Supports loading checkpoints from both old (classifier.0/classifier.1) 
    and new (norm/head) naming conventions.
    """
    
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_dinov3.lvd1689m',
        pretrained: bool = True,
        num_classes: int = 1000,
        img_size: int = 256,
    ):
        super().__init__()
        
        # Load timm DINOv2/DINOv3 model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove head, return features
            img_size=img_size,
        )
        
        # Get embedding dimension
        self.embed_dim = self.backbone.num_features
        
        # Classification head with LayerNorm (using Sequential for compatibility)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )
        
        # Also create references for new naming (for easier access)
        self.norm = self.classifier[0]
        self.head = self.classifier[1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features (class token)
        features = self.backbone(x)
        # Apply normalization and classification
        logits = self.classifier(features)
        return logits


def get_transforms(model_name: str, img_size: int):
    """Get evaluation transforms using timm's data config."""
    data_config = timm.data.resolve_data_config({}, model=model_name)
    
    if img_size:
        data_config['input_size'] = (3, img_size, img_size)
    
    transform = timm.data.create_transform(
        input_size=data_config['input_size'],
        is_training=False,
        mean=data_config['mean'],
        std=data_config['std'],
        interpolation=data_config['interpolation'],
    )
    
    return transform


def load_category_mapping(train_json_path: str) -> Tuple[Dict[int, int], int, Dict[int, str]]:
    """Load category mapping from training JSON.
    
    Returns:
        cat_id_to_label: mapping from category_id to 0-indexed label
        num_classes: total number of classes
        cat_id_to_name: mapping from category_id to category name (for reporting)
    """
    with open(train_json_path, 'r') as f:
        data = json.load(f)
    
    categories = sorted(data['categories'], key=lambda x: x['id'])
    cat_id_to_label = {cat['id']: idx for idx, cat in enumerate(categories)}
    cat_id_to_name = {cat['id']: cat.get('name', f"cat_{cat['id']}") for cat in categories}
    num_classes = len(categories)
    
    print(f"Loaded {num_classes} categories from {train_json_path}")
    
    return cat_id_to_label, num_classes, cat_id_to_name


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())
        return res


def evaluate(model, loader, device):
    """Evaluate model and compute overall + per-category accuracy."""
    model.eval()
    
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_samples = 0
    
    # Per-category tracking
    per_category_correct = defaultdict(int)
    per_category_total = defaultdict(int)
    
    all_predictions = []
    all_labels = []
    all_cat_ids = []
    
    with torch.no_grad():
        for images, labels, cat_ids in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            logits = model(images)
            
            # Compute accuracy
            batch_size = labels.size(0)
            correct_top1, correct_top5 = accuracy(logits, labels, topk=(1, 5))
            
            total_correct_top1 += correct_top1
            total_correct_top5 += correct_top5
            total_samples += batch_size
            
            # Track per-category accuracy
            _, pred = logits.topk(1, 1, True, True)
            pred = pred.squeeze(1)
            
            for i in range(batch_size):
                cat_id = cat_ids[i].item() if isinstance(cat_ids[i], torch.Tensor) else cat_ids[i]
                per_category_total[cat_id] += 1
                if pred[i] == labels[i]:
                    per_category_correct[cat_id] += 1
            
            # Store for confusion matrix if needed
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_cat_ids.extend([c.item() if isinstance(c, torch.Tensor) else c for c in cat_ids])
    
    # Compute overall accuracy
    acc_top1 = 100.0 * total_correct_top1 / total_samples
    acc_top5 = 100.0 * total_correct_top5 / total_samples
    
    return {
        'top1': acc_top1,
        'top5': acc_top5,
        'total_samples': total_samples,
        'per_category_correct': per_category_correct,
        'per_category_total': per_category_total,
        'predictions': all_predictions,
        'labels': all_labels,
        'cat_ids': all_cat_ids,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate DINOv2/DINOv3 on generated subset')
    
    # Model and checkpoint
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to trained model checkpoint')
    parser.add_argument('--model_name', type=str, default=None,
                        help='timm model name (auto-detected from checkpoint if not provided)')
    parser.add_argument('--img_size', type=int, default=None,
                        help='Input image size (auto-detected from checkpoint if not provided)')
    
    # Data
    parser.add_argument('--sample_dir', type=str, required=True,
                        help='Directory with generated images in folder structure')
    parser.add_argument('--train_json', type=str, required=True,
                        help='Training JSON to map category IDs to labels (needs categories field)')
    
    # Evaluation
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Output
    parser.add_argument('--output_json', type=str, default=None,
                        help='Save detailed results to JSON file')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint first to auto-detect model config
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Extract model config from checkpoint args
    ckpt_args = checkpoint.get('args', {})
    
    # Auto-detect or use provided values
    if args.model_name:
        model_name = args.model_name
        print(f"Using provided model name: {model_name}")
    else:
        # Try to get from checkpoint args (key could be 'model', 'model_name', or 'arch')
        model_name = ckpt_args.get('model') or ckpt_args.get('model_name') or ckpt_args.get('arch')
        if model_name:
            print(f"Auto-detected model name from checkpoint: {model_name}")
        else:
            raise ValueError(
                "Could not auto-detect model name from checkpoint. "
                "Please provide --model_name explicitly (e.g., vit_base_patch14_dinov2.lvd142m)"
            )
    
    if args.img_size:
        img_size = args.img_size
        print(f"Using provided image size: {img_size}")
    else:
        # Try to get from checkpoint args
        img_size = ckpt_args.get('img_size')
        if img_size:
            print(f"Auto-detected image size from checkpoint: {img_size}")
        else:
            # Fallback based on model name
            if 'dinov3' in model_name:
                img_size = 256
                print(f"No img_size in checkpoint. Using default for DINOv3: {img_size}")
            else:
                img_size = 224
                print(f"No img_size in checkpoint. Using default for DINOv2: {img_size}")
    
    print(f"\nCheckpoint info:")
    print(f"  - Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  - Model: {model_name}")
    print(f"  - Image size: {img_size}")
    if 'best_acc' in checkpoint:
        print(f"  - Best validation accuracy: {checkpoint['best_acc']:.2f}%")
    
    # Load category mapping from training JSON
    # This is REQUIRED to map folder names (category IDs) to model output indices
    print(f"\nLoading category mapping from {args.train_json}")
    cat_id_to_label, num_classes, cat_id_to_name = load_category_mapping(args.train_json)
    
    # Get num_classes from checkpoint if available, otherwise use train_json
    num_classes_ckpt = ckpt_args.get('num_classes', None)
    
    if num_classes_ckpt and num_classes_ckpt != num_classes:
        print(f"\nWARNING: Checkpoint was trained with {num_classes_ckpt} classes, "
              f"but training JSON has {num_classes} classes!")
        print(f"Using {num_classes_ckpt} from checkpoint for model architecture.")
        num_classes = num_classes_ckpt
    
    # Create model with detected/specified architecture
    print(f"\nCreating model...")
    model = DINOv3Classifier(
        model_name=model_name,
        pretrained=False,  # We'll load trained weights
        num_classes=num_classes,
        img_size=img_size,
    )
    
    # Load weights - handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        raise ValueError("Checkpoint format not recognized. Expected 'state_dict' or 'model_state_dict' key.")
    
    # Load with strict=False to handle naming differences
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Check if loading was successful (classifier weights should be present)
    if missing_keys:
        # Filter out the reference keys (norm/head) which are just aliases
        critical_missing = [k for k in missing_keys if not (k.startswith('norm.') or k.startswith('head.'))]
        if critical_missing:
            print(f"WARNING: Missing keys in checkpoint: {critical_missing}")
    
    model = model.to(device)
    model.eval()
    
    # Create transform using the detected/specified model name and img_size
    transform = get_transforms(model_name, img_size)
    
    # Create dataset
    print(f"\nLoading images from {args.sample_dir}")
    dataset = SubsetImageDataset(
        args.sample_dir,
        cat_id_to_label,
        transform=transform
    )
    
    if len(dataset) == 0:
        print("ERROR: No images found! Check sample_dir and train_json paths.")
        return
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Evaluate
    print(f"\nEvaluating on {len(dataset)} images...")
    results = evaluate(model, loader, device)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total samples: {results['total_samples']}")
    print(f"Top-1 Accuracy: {results['top1']:.2f}%")
    print(f"Top-5 Accuracy: {results['top5']:.2f}%")
    print(f"Categories evaluated: {len(results['per_category_total'])}")
    
    # Per-category accuracy
    print("\n" + "-"*60)
    print("PER-CATEGORY ACCURACY (Top 10 best and worst)")
    print("-"*60)
    
    # Compute per-category accuracy percentages
    per_cat_acc = {}
    for cat_id in results['per_category_total']:
        correct = results['per_category_correct'][cat_id]
        total = results['per_category_total'][cat_id]
        per_cat_acc[cat_id] = 100.0 * correct / total if total > 0 else 0.0
    
    # Sort by accuracy
    sorted_cats = sorted(per_cat_acc.items(), key=lambda x: x[1], reverse=True)
    
    # print("\nBest performing categories:")
    # for cat_id, acc in sorted_cats[:10]:
    #     name = cat_id_to_name.get(cat_id, f"cat_{cat_id}")
    #     total = results['per_category_total'][cat_id]
    #     print(f"  {cat_id:5d} | {acc:6.2f}% | {total:4d} samples | {name[:60]}")
    
    # print("\nWorst performing categories:")
    # for cat_id, acc in sorted_cats[-10:]:
    #     name = cat_id_to_name.get(cat_id, f"cat_{cat_id}")
    #     total = results['per_category_total'][cat_id]
    #     print(f"  {cat_id:5d} | {acc:6.2f}% | {total:4d} samples | {name[:60]}")
    
    # Save detailed results if requested
    if args.output_json:
        output_data = {
            'overall': {
                'top1_accuracy': results['top1'],
                'top5_accuracy': results['top5'],
                'total_samples': results['total_samples'],
                'num_categories': len(results['per_category_total']),
            },
            'per_category': {
                int(cat_id): {
                    'accuracy': per_cat_acc[cat_id],
                    'correct': results['per_category_correct'][cat_id],
                    'total': results['per_category_total'][cat_id],
                    'name': cat_id_to_name.get(cat_id, f"cat_{cat_id}"),
                }
                for cat_id in results['per_category_total']
            },
            'args': vars(args),
            'checkpoint_info': {
                'path': args.checkpoint,
                'epoch': checkpoint.get('epoch', None),
                'best_acc': checkpoint.get('best_acc', None),
                'model_name': model_name,
                'img_size': img_size,
                'num_classes': num_classes,
            }
        }
        
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results saved to: {args.output_json}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
