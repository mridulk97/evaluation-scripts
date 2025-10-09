import open_clip
from PIL import Image
import requests
import torch
import json
import os

import os
import glob
from PIL import Image


import json
import pandas as pd
from tqdm import tqdm

# path = '/users/PAS0536/aminr8/TaxonomyGen/pre_pration/classified_images_final.json'
# path = '../datasets/iNaturalist_2021/val.json'
path = '/users/PAS2136/mridul/ijcv/TaxaDiffusion/TaxaDiffusion/taxa_diffusion/datasets/subsets/subset_dataset_inat_200.json'
# path = '/fs/scratch/PAS0536/amin/iNaturalist_2021/train.json'
# Load the dataset from a JSON file
with open(path) as f:
    data = json.load(f)


# Load the model, preprocessors, and tokenizer
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')


fish_taxonomy = {
    'kingdom': {}
}

fish_path = "/users/PAS2136/mridul/ijcv/TaxaDiffusion/TaxaDiffusion/taxa_diffusion/datasets/fish_data/final.csv"
fish_data = pd.read_csv(fish_path)
for _, row in fish_data.iterrows():
    kingdom = "1"
    phylum = "1"
    class_ = row['Class']
    order = row['Order'] 
    family = row['Family']
    genus = row['Genus']
    species = row['species'] if not pd.isna(row['species']) else 'any'

    if kingdom not in fish_taxonomy['kingdom']:
        fish_taxonomy['kingdom'][kingdom] = {}
    
    if phylum not in fish_taxonomy['kingdom'][kingdom]:
        fish_taxonomy['kingdom'][kingdom][phylum] = {}
    
    if class_ not in fish_taxonomy['kingdom'][kingdom][phylum]:
        fish_taxonomy['kingdom'][kingdom][phylum][class_] = {}
    
    if order not in fish_taxonomy['kingdom'][kingdom][phylum][class_]:
        fish_taxonomy['kingdom'][kingdom][phylum][class_][order] = {}
    
    if family not in fish_taxonomy['kingdom'][kingdom][phylum][class_][order]:
        fish_taxonomy['kingdom'][kingdom][phylum][class_][order][family] = {}
    
    if genus not in fish_taxonomy['kingdom'][kingdom][phylum][class_][order][family]:
        fish_taxonomy['kingdom'][kingdom][phylum][class_][order][family][genus] = []
    
    fish_taxonomy['kingdom'][kingdom][phylum][class_][order][family][genus].append(species)

# Initialize taxonomy mappings
taxonomy = {
    'kingdom': {}
}

# Build the taxonomy hierarchy from the dataset
for category in data['categories']:
    kingdom = category['kingdom']
    phylum = category['phylum']
    class_ = category['class']
    order = category['order']
    family = category['family']
    genus = category['genus']
    species = category['specific_epithet']
    
    if kingdom not in taxonomy['kingdom']:
        taxonomy['kingdom'][kingdom] = {}
    
    if phylum not in taxonomy['kingdom'][kingdom]:
        taxonomy['kingdom'][kingdom][phylum] = {}
    
    if class_ not in taxonomy['kingdom'][kingdom][phylum]:
        taxonomy['kingdom'][kingdom][phylum][class_] = {}
    
    if order not in taxonomy['kingdom'][kingdom][phylum][class_]:
        taxonomy['kingdom'][kingdom][phylum][class_][order] = {}
    
    if family not in taxonomy['kingdom'][kingdom][phylum][class_][order]:
        taxonomy['kingdom'][kingdom][phylum][class_][order][family] = {}
    
    if genus not in taxonomy['kingdom'][kingdom][phylum][class_][order][family]:
        taxonomy['kingdom'][kingdom][phylum][class_][order][family][genus] = []
    
    taxonomy['kingdom'][kingdom][phylum][class_][order][family][genus].append(species)

# The taxonomy structure is now built


def get_keys_at_level(taxonomy, level):
    """
    Given a taxonomic level, return all distinct keys (names) at that level.
    
    Args:
    taxonomy (dict): The full taxonomy dictionary.
    level (str): The taxonomic level to search for (e.g., 'kingdom', 'phylum', 'class', 'order', 'family', 'genus').

    Returns:
    list: A list of unique names at the specified level.
    """
    keys_list = set()  # Use a set to avoid duplicates

    # Traverse the taxonomy tree and collect distinct keys at the given level
    if level == 'kingdom':
        keys_list.update(taxonomy['kingdom'].keys())
    
    elif level == 'phylum':
        for kingdom, phyla in taxonomy['kingdom'].items():
            keys_list.update(phyla.keys())
    
    elif level == 'class':
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                keys_list.update(classes.keys())
    
    elif level == 'order':
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                for class_, orders in classes.items():
                    keys_list.update(orders.keys())
    
    elif level == 'family':
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                for class_, orders in classes.items():
                    for order_, families in orders.items():
                        keys_list.update(families.keys())
    
    elif level == 'genus':
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                for class_, orders in classes.items():
                    for order_, families in orders.items():
                        for family, genera in families.items():
                            keys_list.update(genera.keys())

    elif level == 'specific_epithet' or level == 'species':  # This is the species level
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                for class_, orders in classes.items():
                    for order_, families in orders.items():
                        for family, genera in families.items():
                            for genus, species in genera.items():
                                keys_list.update(species)

    keys_list = {item for item in keys_list if isinstance(item, str)}

    print(keys_list)

    return sorted(keys_list)

# Example usage:
# level = 'specific_epithet'
# families = get_keys_at_level(taxonomy, level)

# print(f"All distinct keys at the {level} level:")
# print(len(families))
# for family in families:
#     print(family)



def get_lineage_bottom_to_top(taxonomy, target_name, level, logging=None):
    # Recursive function to traverse bottom-up from any level to kingdom
    def traverse(kingdom, phylum, class_, order, family, genus, species):
        if species:
            return {
                'species': species,
                'genus': genus,
                'family': family,
                'order': order,
                'class': class_,
                'phylum': phylum,
                'kingdom': kingdom
            }
        if genus:
            return {
                'genus': genus,
                'family': family,
                'order': order,
                'class': class_,
                'phylum': phylum,
                'kingdom': kingdom
            }
        if family:
            return {
                'family': family,
                'order': order,
                'class': class_,
                'phylum': phylum,
                'kingdom': kingdom
            }
        if order:
            return {
                'order': order,
                'class': class_,
                'phylum': phylum,
                'kingdom': kingdom
            }
        if class_:
            return {
                'class': class_,
                'phylum': phylum,
                'kingdom': kingdom
            }
        if phylum:
            return {
                'phylum': phylum,
                'kingdom': kingdom
            }
        if kingdom:
            return {
                'kingdom': kingdom
            }

    # Searching based on level
    if level == 'species':
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                for class_, orders in classes.items():
                    for order_, families in orders.items():
                        for family, genera in families.items():
                            for genus, species_list in genera.items():
                                if target_name in species_list:
                                    return traverse(kingdom, phylum, class_, order_, family, genus, target_name)
    
    elif level == 'genus':
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                for class_, orders in classes.items():
                    for order_, families in orders.items():
                        for family, genera in families.items():
                            if target_name in genera:
                                return traverse(kingdom, phylum, class_, order_, family, target_name, None)
    
    elif level == 'family':
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                for class_, orders in classes.items():
                    for order_, families in orders.items():
                        if target_name in families:
                            return traverse(kingdom, phylum, class_, order_, target_name, None, None)
    
    elif level == 'order':
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                for class_, orders in classes.items():
                    if target_name in orders:
                        return traverse(kingdom, phylum, class_, target_name, None, None, None)
    
    elif level == 'class':
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                if target_name in classes:
                    return traverse(kingdom, phylum, target_name, None, None, None, None)
    
    elif level == 'phylum':
        for kingdom, phyla in taxonomy['kingdom'].items():
            if target_name in phyla:
                return traverse(kingdom, target_name, None, None, None, None, None)
    
    elif level == 'kingdom':
        if target_name in taxonomy['kingdom']:
            return traverse(target_name, None, None, None, None, None, None)
    
    return {}


def load_mappings(mapping_file="condition_mappings.txt"):
    mappings = {
        'kingdom': {}, 'phylum': {}, 'class': {}, 
        'order': {}, 'family': {}, 'genus': {}, 
        'specific_epithet': {}
    }
    with open(mapping_file, 'r') as f:
        current_key = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line in mappings:
                current_key = line
            elif current_key:
                value, index = line.split(': ')
                mappings[current_key][value] = int(index)
    return mappings


def map_condition(category, condition_mappings):
    """Map condition values to integer indices with cascading None behavior."""
    conditions = {}
    conditions_list = []
    for key in condition_mappings.keys():
        if key in category:
            value = category[key]
            conditions[key] = condition_mappings[key].get(value, condition_mappings[key]['None'])
            conditions_list.append(conditions[key])
        else:
            conditions[key] = condition_mappings[key]['None']
            conditions_list.append(conditions[key])
    return conditions, conditions_list

# condition_mappings = load_mappings()
# category = {'genus': 'Ascia', 'family': 'Pieridae', 'order': 'Lepidoptera', 'class': 'Insecta', 'phylum': 'Arthropoda', 'kingdom': 'Animalia'}
# a, b = map_condition(category, condition_mappings)


def get_paths_at_level(taxonomy, level):
    """
    Recursively traverse the taxonomy structure to collect paths up to a given taxonomic level.
    
    Args:
    taxonomy (dict): The full taxonomy dictionary.
    level (int): The taxonomic level to retrieve paths for (1=kingdom, 2=phylum, ..., 7=species).

    Returns:
    list: A list of dictionaries, each representing a full path up to the given level.
    """
    paths = []

    def traverse(node, current_path, current_level):
        # Base case: if we've reached the target level, append the current path
        if current_level == level:
            paths.append(current_path.copy())
            return
        
        # Recursive case: continue traversing the tree
        if current_level == 1:  # We are at the 'kingdom' level
            for kingdom, phyla in node['kingdom'].items():
                traverse(phyla, {**current_path, 'kingdom': kingdom}, current_level + 1)
        
        elif current_level == 2:  # We are at the 'phylum' level
            for phylum, classes in node.items():
                traverse(classes, {**current_path, 'phylum': phylum}, current_level + 1)
        
        elif current_level == 3:  # We are at the 'class' level
            for class_, orders in node.items():
                traverse(orders, {**current_path, 'class': class_}, current_level + 1)
        
        elif current_level == 4:  # We are at the 'order' level
            for order, families in node.items():
                traverse(families, {**current_path, 'order': order}, current_level + 1)
        
        elif current_level == 5:  # We are at the 'family' level
            for family, genera in node.items():
                traverse(genera, {**current_path, 'family': family}, current_level + 1)
        
        elif current_level == 6:  # We are at the 'genus' level
            for genus, species_list in node.items():
                traverse(species_list, {**current_path, 'genus': genus}, current_level + 1)
        
        elif current_level == 7:  # We are at the 'species' level
            for species in node:
                paths.append({**current_path, 'species': species})

    # Start the traversal from the root at level 1 (kingdom)
    traverse(taxonomy, {}, 1)

    return paths


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



# directories = ["/home/karimimonsefi.1/TaxonomyGen/output/v3_new_inat/v3_taxonomy_gen_osc_inf-2025-03-07T02-01-55/samples"]  # List of directories containing folders
# directories = ['/fs/ess/PAS2136/bio_diffusion/samples/baselines/inat/flux1-dev-genus']
# directories = '/fs/ess/PAS2136/bio_diffusion/ip-adapter_runs/samples/samples/taxabind_2gpus_58k_10samples'
directories = '/fs/ess/PAS2136/bio_diffusion/ip-adapter_runs/samples/samples/bioclip_2gpus_58k'
images_dict = load_images_from_directories(directories)

total_score = 0
total_number = 0

# Parse folder name to extract all 7 taxonomy levels
# Expected format: kingdom_phylum_class_order_family_genus_species
for label, images in tqdm(images_dict.items()):

    # print(f"Processing: {label}")
    # print(f"Number of images: {len(images)}")
    
    # Split folder name by underscore to get taxonomy levels
    # Assuming format: kingdom_phylum_class_order_family_genus_species
    parts = label.split("_")
    
    # # Extract taxonomy from folder name (adjust indices based on your folder naming convention)
    # # If your folder has format like "03360_Animalia_Chordata_Aves_..." extract accordingly
    # lineage = {}
    
    # # Try to parse the folder name - you may need to adjust this based on your exact format
    # if len(parts) >= 7:
    #     # If folder name contains all 7 levels directly
    #     lineage = {
    #         'kingdom': parts[1] if len(parts) > 0 else 'None',
    #         'phylum': parts[2] if len(parts) > 1 else 'None',
    #         'class': parts[3] if len(parts) > 2 else 'None',
    #         'order': parts[4] if len(parts) > 3 else 'None',
    #         'family': parts[5] if len(parts) > 4 else 'None',
    #         'genus': parts[6] if len(parts) > 5 else 'None',
    #         'species': parts[7] if len(parts) > 6 else 'None',
    #     }
    # else:
    #     # Try to look up lineage from taxonomy structure
    #     # Assuming the last part is the most specific level
    #     target_name = parts[-1]
    #     # Try different levels from most specific to least specific
    #     for level in ['species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']:
    #         lineage = get_lineage_bottom_to_top(taxonomy, target_name, level, None)
    #         if lineage:
    #             break
    
    # # Build full 7-level taxonomy prompt
    # conditions_list_name = [
    #     f"kingdom: {lineage.get('kingdom', 'None')}",
    #     f"phylum: {lineage.get('phylum', 'None')}",
    #     f"class: {lineage.get('class', 'None')}",
    #     f"order: {lineage.get('order', 'None')}",
    #     f"family: {lineage.get('family', 'None')}",
    #     f"genus: {lineage.get('genus', 'None')}",
    #     f"specific_epithet: {lineage.get('species', 'None')}",
    # ]
    
    # # Create prompt with all 7 taxonomy levels
    # prompt = " ".join(conditions_list_name)

    prompt = ' '.join(parts[1:])
    
    # print(f"Full taxonomy prompt: {prompt}")
    # print("-------")

    for image in images:
        sd_clip_score = calculate_clip_score(image, prompt)
        total_number += 1
        total_score += sd_clip_score


print('Total score:', total_score)
print('Total number of images:', total_number)
print('Average score:', total_score / total_number)
