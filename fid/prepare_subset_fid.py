#!/usr/bin/env python3
"""
Prepare matched real/generated image folders for FID calculation.
Ensures both folders contain exactly the same classes.

Usage:
    python prepare_subset_fid.py --generated /path/to/generated --real_all /path/to/all_real --output_real /path/to/subset_real
"""

import os
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def get_class_folders(folder):
    """Get list of class folder names."""
    classes = []
    for item in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, item)):
            classes.append(item)
    return sorted(classes)


def copy_matching_classes(generated_folder, real_all_folder, output_real_folder, symlink=False):
    """
    Copy/symlink real images for only the classes present in generated folder.
    
    Args:
        generated_folder: Folder with generated images (defines which classes to include)
        real_all_folder: Folder with all real images
        output_real_folder: Output folder for matched real images subset
        symlink: If True, create symlinks instead of copying (saves space)
    """
    # Get classes from generated folder
    generated_classes = get_class_folders(generated_folder)
    print(f"Found {len(generated_classes)} classes in generated folder:")
    for cls in generated_classes[:5]:
        print(f"  - {cls}")
    if len(generated_classes) > 5:
        print(f"  ... and {len(generated_classes) - 5} more")
    
    # Create output directory
    os.makedirs(output_real_folder, exist_ok=True)
    
    # Copy/symlink matching classes from real folder
    print(f"\nCopying/linking matching classes from real folder...")
    matched_count = 0
    missing_classes = []
    
    for class_name in tqdm(generated_classes, desc="Processing classes"):
        real_class_path = os.path.join(real_all_folder, class_name)
        output_class_path = os.path.join(output_real_folder, class_name)
        
        if not os.path.exists(real_class_path):
            missing_classes.append(class_name)
            print(f"\nWarning: Class not found in real folder: {class_name}")
            continue
        
        if symlink:
            # Create symlink (saves disk space)
            if not os.path.exists(output_class_path):
                os.symlink(real_class_path, output_class_path)
        else:
            # Copy directory
            if os.path.exists(output_class_path):
                shutil.rmtree(output_class_path)
            shutil.copytree(real_class_path, output_class_path)
        
        matched_count += 1
    
    print(f"\n✅ Matched {matched_count} classes")
    
    if missing_classes:
        print(f"\n⚠️  Warning: {len(missing_classes)} classes not found in real folder:")
        for cls in missing_classes[:10]:
            print(f"  - {cls}")
        if len(missing_classes) > 10:
            print(f"  ... and {len(missing_classes) - 10} more")
    
    # Count images
    real_image_count = count_images(output_real_folder)
    gen_image_count = count_images(generated_folder)
    
    print(f"\n📊 Statistics:")
    print(f"  Real images (subset): {real_image_count}")
    print(f"  Generated images: {gen_image_count}")
    print(f"  Classes: {matched_count}")
    
    return matched_count, missing_classes


def count_images(folder):
    """Count total images in folder."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.JPG', '.JPEG', '.PNG', '.BMP', '.GIF'}
    count = 0
    # followlinks=True is crucial for symlinked directories!
    for root, dirs, files in os.walk(folder, followlinks=True):
        for f in files:
            if Path(f).suffix in image_extensions:
                count += 1
    return count


def verify_class_match(folder1, folder2):
    """Verify that two folders have the same classes."""
    classes1 = set(get_class_folders(folder1))
    classes2 = set(get_class_folders(folder2))
    
    print(f"\n🔍 Verification:")
    print(f"  Folder 1: {len(classes1)} classes")
    print(f"  Folder 2: {len(classes2)} classes")
    
    only_in_1 = classes1 - classes2
    only_in_2 = classes2 - classes1
    common = classes1 & classes2
    
    print(f"  Common classes: {len(common)}")
    
    if only_in_1:
        print(f"  ⚠️  Only in folder 1: {len(only_in_1)}")
        for cls in list(only_in_1)[:5]:
            print(f"    - {cls}")
    
    if only_in_2:
        print(f"  ⚠️  Only in folder 2: {len(only_in_2)}")
        for cls in list(only_in_2)[:5]:
            print(f"    - {cls}")
    
    if classes1 == classes2:
        print(f"  ✅ Perfect match! Both folders have identical classes.")
        return True
    else:
        print(f"  ⚠️  Mismatch detected. FID score may not be meaningful.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare matched real/generated folders for FID calculation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Copy matching classes (uses disk space but safer)
  python prepare_subset_fid.py \\
    --generated /path/to/generated \\
    --real_all /fs/ess/PAS2136/bio_diffusion/data/inat/images \\
    --output_real /path/to/real_subset

  # Use symlinks (saves disk space, faster)
  python prepare_subset_fid.py \\
    --generated /path/to/generated \\
    --real_all /fs/ess/PAS2136/bio_diffusion/data/inat/images \\
    --output_real /path/to/real_subset \\
    --symlink

  # Verify that two folders have matching classes
  python prepare_subset_fid.py \\
    --verify \\
    --generated /path/to/generated \\
    --real_all /path/to/real_subset
        """
    )
    
    parser.add_argument(
        "--generated",
        type=str,
        required=True,
        help="Folder with generated images (defines which classes to include)"
    )
    parser.add_argument(
        "--real_all",
        type=str,
        required=True,
        help="Folder with all real images"
    )
    parser.add_argument(
        "--output_real",
        type=str,
        help="Output folder for matched real images subset"
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying (saves disk space)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify that folders have matching classes (no copying)"
    )
    
    args = parser.parse_args()
    
    # Validate folders exist
    if not os.path.exists(args.generated):
        print(f"Error: Generated folder not found: {args.generated}")
        return
    if not os.path.exists(args.real_all):
        print(f"Error: Real folder not found: {args.real_all}")
        return
    
    if args.verify:
        # Only verify
        verify_class_match(args.generated, args.real_all)
    else:
        # Copy/symlink matching classes
        if not args.output_real:
            print("Error: --output_real required when not using --verify")
            return
        
        copy_matching_classes(
            args.generated,
            args.real_all,
            args.output_real,
            symlink=args.symlink
        )
        
        # Verify the result
        print(f"\n" + "="*60)
        verify_class_match(args.generated, args.output_real)
        print("="*60)
        
        print(f"\n✅ Done! Now calculate FID:")
        print(f"\npython calculate_fid_from_folders.py \\")
        print(f"  --real {args.output_real} \\")
        print(f"  --generated {args.generated} \\")
        print(f"  --batch-size 64")


if __name__ == "__main__":
    main()
