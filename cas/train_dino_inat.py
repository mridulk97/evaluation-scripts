#!/usr/bin/env python3
"""
Train DINOv2 model on iNaturalist 2021 dataset with 10,000 classes.
Fine-tuning a pre-trained DINOv2 backbone with a classification head.
"""

import os
import time
import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Try to import timm for DINOv2 models
try:
    import timm
    print(f"Using timm version: {timm.__version__}")
except ImportError:
    print("Please install timm: pip install timm")
    exit(1)
from timm.data import resolve_model_data_config, create_transform


class iNat21Dataset(Dataset):
    """
    iNaturalist 2021 dataset loader for COCO JSON format.
    JSON must contain 'images', 'annotations', and 'categories'.
    """
    
    def __init__(self, img_dir, json_path, split='train', transform=None):
        """
        Args:
            img_dir: Base directory where images are stored
            json_path: Path to COCO-style JSON file
            split: 'train' or 'val' (for logging only)
            transform: Image transformations
        """
        self.img_dir = img_dir
        self.split = split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        print(f"\n=== Loading {split} dataset ===")
        print(f"JSON: {json_path}")
        print(f"Image dir: {img_dir}")
        
        if not os.path.exists(json_path):
            raise ValueError(f"JSON file not found: {json_path}")
        if not os.path.exists(img_dir):
            raise ValueError(f"Image directory not found: {img_dir}")
        
        self._load_from_json(json_path)
        
        print(f"✓ Loaded {len(self.samples)} samples")
        print(f"✓ Number of classes: {len(self.class_to_idx)}\n")
    
    def _load_from_json(self, json_path):
        """Load from COCO-style JSON"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Build class mapping from categories (use category_id as key)
        categories = {cat['id']: cat for cat in data['categories']}
        cat_ids = sorted(categories.keys())
        self.class_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
        
        print(f"  Categories: {len(categories)} (ID range: {min(cat_ids)}-{max(cat_ids)})")
        
        # Build image mapping
        images = {img['id']: img for img in data['images']}
        print(f"  Images: {len(images)}")
        print(f"  Annotations: {len(data['annotations'])}")
        
        # Debug: Check first few paths
        debug_count = 0
        
        # Load annotations
        for ann in tqdm(data['annotations'], desc=f"  Loading {self.split} annotations"):
            img_id = ann['image_id']
            cat_id = ann['category_id']
            
            if img_id not in images:
                continue
            
            img_info = images[img_id]
            # Construct full path: img_dir + file_name from JSON
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            
            # Debug first 3 paths
            if debug_count < 3:
                print(f"    Path example: {img_path} (exists: {os.path.exists(img_path)})")
                debug_count += 1
            
            if os.path.exists(img_path):
                class_idx = self.class_to_idx[cat_id]
                self.samples.append((img_path, class_idx))
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid image paths found! Check that img_dir is correct.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image on error
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class DINOv2Classifier(nn.Module):
    """DINOv2 model with classification head"""
    
    def __init__(self, model_name='vit_base_patch14_dinov2.lvd142m', 
                 num_classes=10000, pretrained=True, freeze_backbone=False,
                 img_size=224):
        super().__init__()
        
        print(f"Loading DINOv2 model: {model_name}")
        # Load pre-trained DINOv2 from timm with custom image size
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            img_size=img_size,  # Override default image size
            dynamic_img_size=True,  # Allow flexible input sizes
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        self.input_size = img_size  # Use the specified size
        print(f"Feature dimension: {self.feature_dim}")
        print(f"Using input size: {self.input_size}x{self.input_size} (overridden for faster training)")
        
        # Freeze backbone if requested
        if freeze_backbone:
            print("Freezing backbone parameters")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        print(f"Created classifier with {num_classes} classes")
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        # Classify
        logits = self.classifier(features)
        return logits


def get_transforms_from_model(model, input_size=224, is_train=True):
    """Create transforms matched to the model's expected preprocessing via timm.

    This ensures mean/std, interpolation, and crop settings align with DINOv2.
    We override input_size to the requested fixed size (e.g., 224) for speed.
    """
    data_config = resolve_model_data_config(model)
    # Force desired input size while keeping model-specific mean/std/interp
    data_config['input_size'] = (3, input_size, input_size)
    # Build transform
    return create_transform(**data_config, is_training=is_train)


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Step the scheduler at each batch
        scheduler.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description='Train DINOv2 on iNat-21 (JSON format)')
    
    # Data parameters (JSON format required)
    parser.add_argument('--img_dir', type=str, required=True,
                       help='Base directory where images are stored')
    parser.add_argument('--train_json', type=str, required=True,
                       help='Path to training COCO JSON file')
    parser.add_argument('--val_json', type=str, required=True,
                       help='Path to validation COCO JSON file')
    parser.add_argument('--num_classes', type=int, default=10000,
                       help='Number of classes (default: 10000)')
    
    # Model parameters
    parser.add_argument('--model', type=str, 
                       default='vit_base_patch14_dinov2.lvd142m',
                       choices=['vit_small_patch14_dinov2.lvd142m',
                               'vit_base_patch14_dinov2.lvd142m',
                               'vit_large_patch14_dinov2.lvd142m',
                               'vit_giant_patch14_dinov2.lvd142m',
                               'vit_base_patch16_dinov3.lvd1689m'],
                       help='DINOv2 model variant')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size (default: 224 for faster training)')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone and only train classifier (linear probing)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pre-trained weights')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size per GPU (default: 64)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay (default: 0.05)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Warmup epochs (default: 5)')
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers (default: 8)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create model first to get input size
    print("\n=== Creating Model ===")
    model = DINOv2Classifier(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        img_size=args.img_size
    )
    
    # Get the expected input size from model
    input_size = model.input_size
    print(f"\nUsing input size: {input_size}x{input_size}")
    
    # Create datasets with correct, model-aligned transforms
    print("\n=== Loading Datasets ===")
    try:
        train_transform = get_transforms_from_model(model.backbone, input_size=input_size, is_train=True)
        val_transform = get_transforms_from_model(model.backbone, input_size=input_size, is_train=False)
    except Exception as e:
        print(f"Falling back to basic transforms due to error creating timm transforms: {e}")
        train_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    train_dataset = iNat21Dataset(
        img_dir=args.img_dir,
        json_path=args.train_json,
        split='train',
        transform=train_transform
    )
    
    val_dataset = iNat21Dataset(
        img_dir=args.img_dir,
        json_path=args.val_json,
        split='val',
        transform=val_transform
    )
    
    # Update num_classes if different
    actual_num_classes = len(train_dataset.class_to_idx)
    if actual_num_classes != args.num_classes:
        print(f"Updating num_classes from {args.num_classes} to {actual_num_classes}")
        args.num_classes = actual_num_classes
        # Recreate model with correct num_classes
        print("Recreating model with correct num_classes...")
        model = DINOv2Classifier(
            model_name=args.model,
            num_classes=actual_num_classes,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
            img_size=args.img_size
        )
    
    model = model.to(device)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # Model already created above, just print stats
    print("\n=== Model Statistics ===")
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # Optimize only classifier when freezing backbone
    if args.freeze_backbone:
        print("Optimizer configured to update classifier only (frozen backbone)")
        optimizer = optim.AdamW(
            model.classifier.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        # Use parameter groups to give head a slightly higher LR
        head_params = list(model.classifier.parameters())
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            [
                { 'params': backbone_params, 'lr': args.lr },
                { 'params': head_params, 'lr': args.lr * 10.0 },
            ],
            weight_decay=args.weight_decay
        )
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.2f}%")
    
    # Training loop
    print("\n=== Starting Training ===")
    print(f"Training for {args.epochs} epochs")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Steps per epoch: {len(train_loader)}")
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_acc': best_acc,
            'args': vars(args)
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint_latest.pth'))
        
        # Save best checkpoint
        if is_best:
            print(f"  New best accuracy: {val_acc:.2f}%")
            torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint_best.pth'))
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, 
                      os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    total_time = time.time() - training_start_time
    print(f"\n=== Training Complete ===")
    print(f"Total training time: {total_time / 3600:.2f} hours")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
