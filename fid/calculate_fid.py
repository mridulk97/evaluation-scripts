#!/usr/bin/env python3
"""
Calculate FID score with automatic cleanup and proper image resizing.
Works with nested folder structures and handles variable image sizes.
"""

import argparse
import torch
import os
import shutil
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def create_flat_structure(source_folder, temp_folder):
    """
    Create a flat structure of images (all in one folder) from nested structure.
    Uses symlinks to avoid copying files.
    """
    print(f"Creating flat structure with symlinks...")
    os.makedirs(temp_folder, exist_ok=True)
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.JPG', '.JPEG', '.PNG', '.BMP', '.GIF', '.webp', '.tif', '.tiff')
    count = 0
    
    # Walk through source folder and symlink all images to temp folder
    for root, dirs, files in os.walk(source_folder, followlinks=True):
        for filename in files:
            if filename.endswith(image_extensions):
                source_path = os.path.join(root, filename)
                # Create unique filename to avoid conflicts
                rel_path = os.path.relpath(source_path, source_folder)
                safe_name = rel_path.replace('/', '_').replace('\\', '_')
                dest_path = os.path.join(temp_folder, safe_name)
                
                # Create symlink
                try:
                    os.symlink(source_path, dest_path)
                    count += 1
                except Exception as e:
                    print(f"Warning: Could not symlink {filename}: {e}")
    
    print(f"  Created {count} symlinks")
    return count


def get_activations_fixed(files, model, batch_size=50, dims=2048, device='cpu', num_workers=1, resize=299):
    """
    Modified version of pytorch-fid's get_activations that properly resizes images.
    This fixes the issue with variable-sized images.
    """
    import torchvision.transforms as TF
    from torch.nn.functional import adaptive_avg_pool2d
    
    model.eval()

    if batch_size > len(files):
        print(f'Warning: batch size ({batch_size}) is bigger than the data size ({len(files)}). '
              f'Setting batch size to data size')
        batch_size = len(files)

    # Custom transform that resizes AND converts to tensor
    transform = TF.Compose([
        TF.Resize((resize, resize)),  # Resize to fixed size (default 299 for InceptionV3)
        TF.ToTensor()
    ])
    
    class ImagePathDataset(torch.utils.data.Dataset):
        def __init__(self, files, transforms=None):
            self.files = files
            self.transforms = transforms

        def __len__(self):
            return len(self.files)

        def __getitem__(self, i):
            path = self.files[i]
            img = Image.open(path).convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img)
            return img

    dataset = ImagePathDataset(files, transforms=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    pred_arr = np.empty((len(files), dims))
    start_idx = 0

    for batch in tqdm(dataloader, desc="Processing batches"):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics_fixed(files, model, batch_size=50, dims=2048, device='cpu', num_workers=1):
    """Calculate mean and covariance of activations."""
    act = get_activations_fixed(files, model, batch_size, dims, device, num_workers, resize=512)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid_with_cleanup(real_folder, generated_folder, batch_size=50, device='cuda', num_workers=4):
    """
    Calculate FID score with automatic cleanup and proper image resizing.
    """
    try:
        from pytorch_fid.inception import InceptionV3
        from scipy import linalg
    except ImportError as e:
        print(f"Error: Required library not installed: {e}")
        print("Install with: pip install pytorch-fid scipy")
        return None
    
    # Create temporary directories for flat structures
    temp_dir = tempfile.mkdtemp(prefix='fid_temp_')
    real_temp = os.path.join(temp_dir, 'real')
    gen_temp = os.path.join(temp_dir, 'generated')
    
    try:
        print(f"\n{'='*60}")
        print("Preparing datasets for FID calculation...")
        print('='*60)
        
        # Create flat structures
        print(f"\n1. Processing real images from: {real_folder}")
        real_count = create_flat_structure(real_folder, real_temp)
        
        print(f"\n2. Processing generated images from: {generated_folder}")
        gen_count = create_flat_structure(generated_folder, gen_temp)
        
        if real_count == 0 or gen_count == 0:
            print("\n❌ Error: One of the folders has no images!")
            return None
        
        # Adjust batch size
        min_images = min(real_count, gen_count)
        if batch_size > min_images:
            batch_size = max(1, min(50, min_images))
            print(f"\n⚠️  Adjusted batch size to {batch_size}")
        
        # Warn about small datasets
        if min_images < 2000:
            print(f"\n⚠️  WARNING: Small dataset ({min_images} images)")
            print(f"   FID scores are more reliable with 2000+ images per distribution")
        
        print(f"\n{'='*60}")
        print("Calculating FID score...")
        print('='*60)
        print(f"Real images: {real_count}")
        print(f"Generated images: {gen_count}")
        print(f"Device: {device}")
        print(f"Batch size: {batch_size}")
        print(f"Num workers: {num_workers}")
        print()
        
        # Load InceptionV3 model
        print("Loading InceptionV3 model...")
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx]).to(device)
        model.eval()
        
        # Get list of files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.JPG', '.JPEG', '.PNG', '.BMP', '.GIF', '.webp', '.tif', '.tiff'}
        
        print("\nCollecting file lists...")
        real_files = sorted([os.path.join(real_temp, f) for f in os.listdir(real_temp) 
                            if any(f.endswith(ext) for ext in image_extensions)])
        gen_files = sorted([os.path.join(gen_temp, f) for f in os.listdir(gen_temp) 
                           if any(f.endswith(ext) for ext in image_extensions)])
        
        print(f"Real files found: {len(real_files)}")
        print(f"Generated files found: {len(gen_files)}")
        
        # Calculate statistics
        print("\nComputing statistics for real images...")
        m1, s1 = calculate_activation_statistics_fixed(
            real_files, model, batch_size, 2048, device, num_workers
        )
        
        print("\nComputing statistics for generated images...")
        m2, s2 = calculate_activation_statistics_fixed(
            gen_files, model, batch_size, 2048, device, num_workers
        )
        
        # Calculate Fréchet distance
        print("\nCalculating Fréchet distance...")
        
        mu1 = np.atleast_1d(m1)
        mu2 = np.atleast_1d(m2)
        sigma1 = np.atleast_2d(s1)
        sigma2 = np.atleast_2d(s2)
        
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if not np.isfinite(covmean).all():
            print("Warning: Product is singular, adding epsilon to diagonal")
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f'Imaginary component {m}')
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        fid_value = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        
        return fid_value
        
    except Exception as e:
        print(f"\n❌ Error during FID calculation: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Clean up temporary directory
        print(f"\nCleaning up temporary directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
            print("✅ Cleanup complete")
        except Exception as e:
            print(f"⚠️  Warning: Could not remove temp directory: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate FID score from nested folder structures with variable image sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calculate_fid_clean.py \\
    --real /fs/ess/PAS2136/bio_diffusion/data/inat/images/train_subset \\
    --generated /path/to/generated \\
    --batch-size 50

Notes:
  - Automatically handles nested folder structures (class_folder/images)
  - Handles variable image sizes (resizes all to 299x299 for InceptionV3)
  - Creates temporary flat structures using symlinks (fast, no copying)
  - Automatically cleans up temporary directories after calculation
  - Works with symlinked class folders
        """
    )
    
    parser.add_argument(
        "--real",
        type=str,
        required=True,
        help="Path to folder with real images (can be nested)"
    )
    parser.add_argument(
        "--generated",
        type=str,
        required=True,
        help="Path to folder with generated images (can be nested)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for FID calculation (default: 50)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help="Device to use (default: cuda if available)"
    )
    
    args = parser.parse_args()
    
    # Validate folders exist
    if not os.path.exists(args.real):
        print(f"Error: Real images folder not found: {args.real}")
        return
    if not os.path.exists(args.generated):
        print(f"Error: Generated images folder not found: {args.generated}")
        return
    
    # Calculate FID with cleanup
    fid_value = calculate_fid_with_cleanup(
        args.real,
        args.generated,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers
    )
    
    if fid_value is not None:
        print(f"\n" + "="*60)
        print(f"FID Score: {fid_value:.4f}")
        print("="*60)
        print(f"\nInterpretation (lower is better):")
        print(f"  < 10  : Excellent - very similar to real images")
        print(f"  10-20 : Very good - high quality generation")
        print(f"  20-50 : Good - decent quality")
        print(f"  50-100: Fair - noticeable differences")
        print(f"  > 100 : Poor - significant quality issues")


if __name__ == "__main__":
    main()
