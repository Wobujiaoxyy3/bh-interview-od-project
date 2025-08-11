"""
Analyze class combination distribution in the dataset
"""

import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def analyze_class_combinations(annotations_file):
    """Analyze class combination distribution"""
    
    # Load COCO format annotation file
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Get category information
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print("Dataset category information:")
    for cat_id, cat_name in categories.items():
        print(f"  Category {cat_id}: {cat_name}")
    print()
    
    # Group annotations by image
    image_annotations = defaultdict(list)
    for annotation in coco_data['annotations']:
        image_annotations[annotation['image_id']].append(annotation)
    
    # Analyze class combinations for each image
    class_combinations = []
    combination_details = defaultdict(list)
    
    for image_id, annotations in image_annotations.items():
        # Get unique classes in this image
        class_ids = sorted(list(set(ann['category_id'] for ann in annotations)))
        combination_key = "_".join(map(str, class_ids))
        
        class_combinations.append(combination_key)
        combination_details[combination_key].append({
            'image_id': image_id,
            'num_annotations': len(annotations),
            'class_names': [categories[cid] for cid in class_ids]
        })
    
    # Count combination distribution
    combination_counts = Counter(class_combinations)
    
    print("=== Class Combination Distribution Analysis ===")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Number of class combinations found: {len(combination_counts)}")
    print()
    
    print("Class combination details:")
    for combination_key, count in sorted(combination_counts.items(), key=lambda x: x[1], reverse=True):
        # Parse class IDs
        class_ids = [int(cid) for cid in combination_key.split('_')]
        class_names = [categories[cid] for cid in class_ids]
        
        print(f"\nCombination '{combination_key}' ({'+'.join(class_names)}): {count} images")
        
        # Show some examples
        examples = combination_details[combination_key][:3]  # Show first 3 examples
        for i, example in enumerate(examples):
            print(f"  Example {i+1}: Image ID {example['image_id']}, {example['num_annotations']} annotations")
        
        if len(examples) < count:
            print(f"  ... (and {count - len(examples)} more similar images)")
    
    print("\n=== Stratified Split Feasibility Analysis ===")
    
    # Check single sample combinations
    single_sample_combinations = [key for key, count in combination_counts.items() if count == 1]
    if single_sample_combinations:
        print(f"Combinations with only 1 sample ({len(single_sample_combinations)} types):")
        for key in single_sample_combinations:
            class_ids = [int(cid) for cid in key.split('_')]
            class_names = [categories[cid] for cid in class_ids]
            print(f"  '{key}' ({'+'.join(class_names)})")
    
    # Check rare combinations (less than 3 samples)
    rare_combinations = [key for key, count in combination_counts.items() if count < 3]
    if rare_combinations:
        print(f"\nRare combinations (<3 samples, {len(rare_combinations)} types):")
        for key in rare_combinations:
            count = combination_counts[key]
            class_ids = [int(cid) for cid in key.split('_')]
            class_names = [categories[cid] for cid in class_ids]
            print(f"  '{key}' ({'+'.join(class_names)}): {count} samples")
    
    # Calculate total rare sample count
    rare_sample_count = sum(count for key, count in combination_counts.items() if count < 3)
    common_sample_count = sum(count for key, count in combination_counts.items() if count >= 3)
    
    print(f"\nSplit strategy recommendations:")
    print(f"  Common samples (>=3): {common_sample_count} images - use stratified split")
    print(f"  Rare samples (<3): {rare_sample_count} images - use random split")
    print(f"  Stratified split coverage: {common_sample_count/len(class_combinations)*100:.1f}%")


if __name__ == "__main__":
    annotations_file = Path("data/processed/2.0-coco_annotations_cleaned.json")
    
    if annotations_file.exists():
        print("Analyzing floor plan dataset class combinations...\n")
        analyze_class_combinations(annotations_file)
    else:
        print(f"Annotation file not found: {annotations_file}")
        print("Please check if the file path is correct")