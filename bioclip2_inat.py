import open_clip
from PIL import Image
import torch
import json
import os
import glob
import pandas as pd
from tqdm import tqdm
import argparse

import argparse

# Global model variables - will be loaded in main()
model = None
preprocess_train = None
preprocess_val = None
tokenizer = None
device = None


def load_images_from_directories(directory):
    # Dictionary to store images with labels
    images_dict = {}


    # Find all folders in the current directory that match the pattern
    for folder_name in tqdm(os.listdir(directory)):
        # Check if the folder name matches the pattern "sample_target_name_{label}"
        # if "sample_target_name_" in folder_name:
        if "" in folder_name:
            # Extract the label from the folder name
            # label = folder_name.split("_")[-1]
            # label = label.replace(" ", "/")

            # label = folder_name.replace("_", " ")
            label = folder_name
            
            # Create the full path to the folder
            # folder_path = os.path.join(directory, folder_name, 'level_4')
            folder_path = os.path.join(directory, folder_name,)
            
            # Check if it's a directory
            if os.path.isdir(folder_path):
                # Initialize list for this label if it doesn't exist
                if label not in images_dict:
                    images_dict[label] = []
                
                # Load all images with common extensions in this folder
                for img_path in glob.glob(os.path.join(folder_path, "*")):
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                        try:
                            image = Image.open(img_path).convert('RGB')  # Load and convert to RGB if necessary
                            images_dict[label].append(image)
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")

    return images_dict


def calculate_clip_score_batch(images, prompts, batch_size=32):
    """
    Calculate CLIP scores for a batch of images and prompts.
    More efficient than processing one at a time.
    """
    scores = []
    
    # Process in batches
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        batch_prompts = prompts[i:i + batch_size]
        
        # Preprocess images and stack them
        image_tensors = torch.stack([preprocess_val(img) for img in batch_images]).to(device)
        text_tokens = tokenizer(batch_prompts).to(device)
        
        # Encode image and text features and normalize
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image_tensors)
            text_features = model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity scores (diagonal for paired samples)
            batch_scores = (image_features * text_features).sum(dim=-1) * 100
            scores.extend(batch_scores.cpu().tolist())
    
    return scores


def main():
    parser = argparse.ArgumentParser(description='Evaluate BioCLIP scores for generated images')
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='Directory containing label subfolders with images')
    parser.add_argument('--output_file', '-o', type=str, default=None,
                        help='Optional CSV file to save detailed results')
    parser.add_argument('--bioclip_version', '-b', type=str, default='bioclip-2',
                        help='Version of BioCLIP to use (default: bioclip-2)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing images (default: 32)')
    args = parser.parse_args()

    # Set up device
    global model, preprocess_train, preprocess_val, tokenizer, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the model, preprocessors, and tokenizer
    if args.bioclip_version == 'bioclip-2':
        print("Loading BioCLIP - 2 model...")
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
        tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')
    else:
        print("Loading BioCLIP - 1 model...")
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
    
    # Move model to GPU
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!")

    # Load images from directories
    print(f"\nLoading images from: {args.input_dir}")
    images_dict = load_images_from_directories(args.input_dir)

    total_score = 0
    total_number = 0
    results = []

    # Process images in batches for efficiency
    print("\nProcessing images...")
    for label, images in tqdm(images_dict.items()):
        parts = label.split("_")
        prompt = ' '.join(parts[1:])
        
        # Create a list of prompts (same prompt for all images in this label)
        prompts = [prompt] * len(images)
        
        # Calculate scores in batches
        scores = calculate_clip_score_batch(images, prompts, batch_size=args.batch_size)
        
        # Accumulate results
        total_number += len(scores)
        total_score += sum(scores)
        
        if args.output_file:
            for score in scores:
                results.append({
                    'label': label,
                    'prompt': prompt,
                    'score': round(score, 4)
                })

    # Print summary
    print('\n' + '='*50)
    print('RESULTS SUMMARY')
    print('='*50)
    print(f'Total score: {total_score}')
    print(f'Total number of images: {total_number}')
    print(f'Average score: {total_score / total_number:.4f}')
    print('='*50)

    # Save detailed results if output file specified
    if args.output_file:
        df = pd.DataFrame(results)
        df.to_csv(args.output_file, index=False)
        print(f'\nDetailed results saved to: {args.output_file}')


if __name__ == '__main__':
    main()

# Example usage:
# python evaluation-scripts/bioclip2_inat.py --input_dir /path/to/generated/images