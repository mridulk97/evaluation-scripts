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


def calculate_clip_score(image, prompt):
    image = preprocess_val(image).unsqueeze(0)
    text = tokenizer([prompt])

    # Encode image and text features and normalize
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute the similarity score for the single prompt
        score = (image_features @ text_features.T).item()
    
    # print(f"Cosine similarity score for the prompt '{prompt}':", score)
    return round(float(score * 100), 4)


def main():
    parser = argparse.ArgumentParser(description='Evaluate BioCLIP scores for generated images')
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='Directory containing label subfolders with images')
    parser.add_argument('--output_file', '-o', type=str, default=None,
                        help='Optional CSV file to save detailed results')
    parser.add_argument('--bioclip_version', '-b', type=str, default='bioclip-2',
                        help='Version of BioCLIP to use (default: bioclip-2)')
    args = parser.parse_args()

    # Load the model, preprocessors, and tokenizer
    global model, preprocess_train, preprocess_val, tokenizer
    if args.bioclip_version == 'bioclip-2':
        print("Loading BioCLIP - 2 model...")
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
        tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')
    else:
        print("Loading BioCLIP - 1 model...")
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
    print("Model loaded successfully!")

    # Load images from directories
    print(f"\nLoading images from: {args.input_dir}")
    images_dict = load_images_from_directories(args.input_dir)

    total_score = 0
    total_number = 0
    results = []

    # Parse folder name to extract all 7 taxonomy levels
    # Expected format: kingdom_phylum_class_order_family_genus_species
    for label, images in tqdm(images_dict.items()):
        parts = label.split("_")
        prompt = ' '.join(parts[1:])
        
        for image in images:
            sd_clip_score = calculate_clip_score(image, prompt)
            total_number += 1
            total_score += sd_clip_score
            
            if args.output_file:
                results.append({
                    'label': label,
                    'prompt': prompt,
                    'score': sd_clip_score
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