import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import argparse
import os
from PIL import Image
from tqdm import tqdm
from datetime import datetime

class ClassBasedPairedImageDataset(Dataset):
    def __init__(self, real_image_folder, generated_image_folder, level, inat_dataset=True):
        # Define transformations: resize and normalize images to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

        print("Loading images and pairing by class...")
        # Group images by class for each folder
        self.generated_images_by_class = self._load_images_by_class(generated_image_folder)
        generated_classes = list(self.generated_images_by_class.keys())
        if inat_dataset:
            self.real_images_by_class = self._load_images_by_class(real_image_folder)
        else:
            self.real_images_by_class = self._load_images_by_class_real(real_image_folder, generated_classes, level)

        # Create paired indices based on class
        self.paired_images = []
        # for class_name in self.real_images_by_class.keys():
            # assert class_name in self.generated_images_by_class
        for class_name in self.generated_images_by_class.keys():
            assert class_name in self.real_images_by_class

            real_images = self.real_images_by_class[class_name]
            generated_images = self.generated_images_by_class[class_name]
            
            # Pair images within the same class up to the minimum number in either folder
            num_pairs = min(len(real_images), len(generated_images))
            for i in range(num_pairs):
                self.paired_images.append((real_images[i], generated_images[i]))

    def _load_images_by_class(self, folder_path):
        images_by_class = {}
        for class_name in os.listdir(folder_path):
            class_path = os.path.join(folder_path, class_name)
            if os.path.isdir(class_path):
                images_by_class[class_name] = [
                    os.path.join(class_path, img_name) for img_name in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, img_name))
                ]
        return images_by_class

    def _load_images_by_class_real(self, folder_path, generated_classes, level):

        images_by_class = {}
        for class_name in os.listdir(folder_path):
            class_path = os.path.join(folder_path, class_name)
            class_level_name = class_name[6:]
            class_level_name = class_level_name.split('_')[:level+1]
            class_level_name = "_".join(class_level_name)
            if os.path.isdir(class_path) and class_level_name in generated_classes:
                images_by_class[class_level_name] = [
                    os.path.join(class_path, img_name) for img_name in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, img_name))
                ]
        return images_by_class

    def __len__(self):
        return len(self.paired_images)

    def __getitem__(self, idx):
        real_image_path, generated_image_path = self.paired_images[idx]
        
        real_image = Image.open(real_image_path).convert('RGB')
        generated_image = Image.open(generated_image_path).convert('RGB')
        
        # Apply transformations
        real_image = self.transform(real_image)
        generated_image = self.transform(generated_image)
        
        return real_image, generated_image

def compute_lpips_batched(real_image_folder, generated_image_folder, net, batch_size, level):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize LPIPS metric using torchmetrics
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type=net).to(device)
    dataset = ClassBasedPairedImageDataset(real_image_folder, generated_image_folder, level)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_lpips_score = 0.0
    num_batches = 0

    for real_images, generated_images in tqdm(dataloader):
        real_images = real_images.to(device)
        generated_images = generated_images.to(device)

        with torch.no_grad():
            batch_score = lpips_metric(real_images, generated_images)
            total_lpips_score += batch_score.item()  # Accumulate the batch score
            num_batches += 1  # Count the batch

    # Calculate average LPIPS score
    average_lpips = total_lpips_score / num_batches if num_batches > 0 else None
    print(f"\nAverage LPIPS Score: {average_lpips}")

    results = {
        "average_lpips": average_lpips
    }

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute LPIPS between real and generated images (batched)")
    parser.add_argument('--real_image_folder', type=str, required=True, help="Path to folder with real images")
    parser.add_argument('--generated_image_folder', type=str, required=True, help="Path to folder with generated images")
    parser.add_argument('--net', type=str, default='vgg', choices=['vgg', 'alex'], help="Backbone network for LPIPS (default: vgg)")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for LPIPS computation (default: 16)")
    parser.add_argument('--output_dir', type=str, default='./metrics/', help="Directory to save LPIPS results as JSON")
    parser.add_argument('--level', type=int, default=6, help="Level of detail for LPIPS computation (default: 6)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results = compute_lpips_batched(args.real_image_folder, args.generated_image_folder, args.net, args.batch_size, args.level)

    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(args.output_dir, f"lpips_results_model_{args.net}_{timestamp}.json")

    # Save LPIPS results to a JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")
