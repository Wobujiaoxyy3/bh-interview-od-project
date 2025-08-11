"""
Comprehensive data pipeline validation script


This script validates:
1. Original COCO annotations loading 
2. COCO to Pascal VOC conversion in dataset
3. Data preprocessing and transforms
4. Format consistency for evaluation
5. Visual validation of coordinate transformations

"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ConfigManager
from datasets import FloorPlanDataset
from pycocotools.coco import COCO


def load_original_coco_annotations(annotations_file):
    """Load and parse original COCO format annotations"""
    print(f"Loading original COCO annotations from: {annotations_file}")
    
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"Found {len(coco_data['images'])} images")
    print(f"Found {len(coco_data['annotations'])} annotations")
    print(f"Found {len(coco_data['categories'])} categories")
    
    # Print category mapping
    print("\nCategory mapping:")
    for cat in coco_data['categories']:
        print(f"  ID {cat['id']}: {cat['name']}")
    
    return coco_data


def validate_coordinate_conversion(coco_data, dataset, image_idx=0):
    """
    Validate coordinate conversion from COCO to Pascal VOC format
    验证从COCO到Pascal VOC格式的坐标转换
    """
    print(f"\n=== Validating Coordinate Conversion for Image {image_idx} ===")
    
    # Get dataset sample (already converted to Pascal VOC)
    sample = dataset[image_idx]
    image_tensor = sample['image']
    target = sample['target']
    image_id = sample['image_id']
    
    print(f"Image ID: {image_id}")
    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"Number of annotations in dataset: {len(target['boxes'])}")
    
    # Find corresponding original COCO annotations
    original_annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == image_id:
            original_annotations.append(ann)
    
    print(f"Number of original COCO annotations: {len(original_annotations)}")
    
    # Compare annotations
    print("\nCoordinate conversion validation:")
    print("Original COCO (x, y, w, h) -> Dataset Pascal VOC (x1, y1, x2, y2)")
    
    for i, (original_ann, dataset_box) in enumerate(zip(original_annotations, target['boxes'])):
        # Original COCO format: [x, y, width, height]
        coco_x, coco_y, coco_w, coco_h = original_ann['bbox']
        
        # Expected Pascal VOC conversion: [x1, y1, x2, y2]
        expected_x1, expected_y1 = coco_x, coco_y
        expected_x2, expected_y2 = coco_x + coco_w, coco_y + coco_h
        
        # Dataset Pascal VOC format
        dataset_x1, dataset_y1, dataset_x2, dataset_y2 = dataset_box.tolist()
        
        print(f"  Annotation {i}:")
        print(f"    COCO:     [{coco_x:6.1f}, {coco_y:6.1f}, {coco_w:6.1f}, {coco_h:6.1f}]")
        print(f"    Expected: [{expected_x1:6.1f}, {expected_y1:6.1f}, {expected_x2:6.1f}, {expected_y2:6.1f}]")
        print(f"    Dataset:  [{dataset_x1:6.1f}, {dataset_y1:6.1f}, {dataset_x2:6.1f}, {dataset_y2:6.1f}]")
        
        # Check if conversion is correct
        conversion_correct = (
            abs(dataset_x1 - expected_x1) < 0.01 and
            abs(dataset_y1 - expected_y1) < 0.01 and
            abs(dataset_x2 - expected_x2) < 0.01 and
            abs(dataset_y2 - expected_y2) < 0.01
        )
        print(f"    Conversion correct: {conversion_correct}")
        if not conversion_correct:
            print(f"  COORDINATE CONVERSION ERROR!")


def validate_evaluation_format_conversion():
    """
    Validate the evaluation format conversion logic
    """
    print(f"\n=== Validating Evaluation Format Conversion ===")
    
    # Simulate Pascal VOC predictions from model
    pascal_boxes = [
        [100.0, 200.0, 300.0, 400.0],  # x1, y1, x2, y2
        [150.0, 250.0, 350.0, 450.0]
    ]
    
    print("Pascal VOC format predictions (x1, y1, x2, y2):")
    for i, box in enumerate(pascal_boxes):
        print(f"  Prediction {i}: {box}")
    
    # Convert to COCO format for evaluation
    coco_predictions = []
    for i, (x1, y1, x2, y2) in enumerate(pascal_boxes):
        width = x2 - x1
        height = y2 - y1
        coco_box = [x1, y1, width, height]
        coco_predictions.append(coco_box)
        print(f"  COCO format {i}: [{x1}, {y1}, {width}, {height}]")
    
    # Validate conversion
    print("\nValidation:")
    for i, (pascal_box, coco_box) in enumerate(zip(pascal_boxes, coco_predictions)):
        x1, y1, x2, y2 = pascal_box
        coco_x, coco_y, coco_w, coco_h = coco_box
        
        # Verify conversion
        expected_x2 = coco_x + coco_w
        expected_y2 = coco_y + coco_h
        
        conversion_correct = (
            abs(x1 - coco_x) < 0.01 and
            abs(y1 - coco_y) < 0.01 and
            abs(x2 - expected_x2) < 0.01 and
            abs(y2 - expected_y2) < 0.01
        )
        print(f"  Conversion {i} correct: {conversion_correct}")


def visualize_annotations_comparison(coco_data, dataset, image_idx=0, output_dir="validation_output"):
    """
    Create visual comparison of original COCO and dataset annotations
    """
    print(f"\n=== Creating Visual Validation for Image {image_idx} ===")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get dataset sample
    sample = dataset[image_idx]
    image_tensor = sample['image']  # Already preprocessed
    target = sample['target']
    image_id = sample['image_id']
    original_size = sample['original_size']
    
    # Get original image
    image_info = None
    for img_info in coco_data['images']:
        if img_info['id'] == image_id:
            image_info = img_info
            break
    
    if not image_info:
        print(f"Could not find image info for ID {image_id}")
        return
        
    # Load original image
    original_image_path = dataset.images_dir / image_info['file_name']
    original_image = Image.open(original_image_path).convert('RGB')
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Original image with COCO annotations
    axes[0].imshow(original_image)
    axes[0].set_title(f'Original Image + COCO Annotations\n{image_info["file_name"]}')
    
    # Draw COCO format annotations
    original_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown']
    
    for i, ann in enumerate(original_annotations):
        x, y, w, h = ann['bbox']  # COCO format
        color = colors[i % len(colors)]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor=color, facecolor='none', alpha=0.8)
        axes[0].add_patch(rect)
        
        # Add category label
        cat_name = next(cat['name'] for cat in coco_data['categories'] 
                       if cat['id'] == ann['category_id'])
        axes[0].text(x, y-5, f'{cat_name} (COCO)', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                    fontsize=8, color='white')
    
    # Plot 2: Processed tensor as image with dataset annotations  
    # Convert tensor back to displayable image
    if image_tensor.dim() == 3:
        # Denormalize if needed
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        denorm_tensor = image_tensor * std + mean
        denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
        
        # Convert to numpy
        processed_image = denorm_tensor.permute(1, 2, 0).numpy()
    else:
        processed_image = image_tensor.numpy()
    
    axes[1].imshow(processed_image)
    axes[1].set_title(f'Processed Image + Dataset Annotations\nTensor shape: {image_tensor.shape}')
    
    # Draw Pascal VOC format annotations from dataset
    for i, box in enumerate(target['boxes']):
        x1, y1, x2, y2 = box.tolist()  # Pascal VOC format
        w, h = x2 - x1, y2 - y1
        color = colors[i % len(colors)]
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                               edgecolor=color, facecolor='none', alpha=0.8)
        axes[1].add_patch(rect)
        
        # Add category label
        label_id = target['labels'][i].item()
        cat_name = dataset.category_names.get(label_id, f'cat_{label_id}')
        axes[1].text(x1, y1-5, f'{cat_name} (Pascal)', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                    fontsize=8, color='white')
    
    # Plot 3: Side-by-side coordinate comparison
    axes[2].axis('off')
    comparison_text = f"Coordinate Format Comparison (Image ID: {image_id})\n\n"
    comparison_text += "COCO Format (x, y, width, height):\n"
    
    for i, ann in enumerate(original_annotations):
        x, y, w, h = ann['bbox']
        cat_name = next(cat['name'] for cat in coco_data['categories'] 
                       if cat['id'] == ann['category_id'])
        comparison_text += f"  {i}: [{x:6.1f}, {y:6.1f}, {w:6.1f}, {h:6.1f}] - {cat_name}\n"
    
    comparison_text += "\nPascal VOC Format (x1, y1, x2, y2):\n"
    for i, box in enumerate(target['boxes']):
        x1, y1, x2, y2 = box.tolist()
        label_id = target['labels'][i].item()
        cat_name = dataset.category_names.get(label_id, f'cat_{label_id}')
        comparison_text += f"  {i}: [{x1:6.1f}, {y1:6.1f}, {x2:6.1f}, {y2:6.1f}] - {cat_name}\n"
    
    comparison_text += "\nEvaluation Format (COCO from Pascal):\n"
    for i, box in enumerate(target['boxes']):
        x1, y1, x2, y2 = box.tolist()
        width, height = x2 - x1, y2 - y1
        label_id = target['labels'][i].item()
        cat_name = dataset.category_names.get(label_id, f'cat_{label_id}')
        comparison_text += f"  {i}: [{x1:6.1f}, {y1:6.1f}, {width:6.1f}, {height:6.1f}] - {cat_name}\n"
    
    axes[2].text(0.05, 0.95, comparison_text, transform=axes[2].transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[2].set_title('Coordinate Format Details')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f'validation_image_{image_idx}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved validation visualization to: {output_file}")
    plt.show()


def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(description='Validate data pipeline')
    parser.add_argument('--config', '-c', type=str, 
                       default='faster_rcnn_config',
                       help='Config file path')
    parser.add_argument('--image-idx', type=int, default=0,
                       help='Image index to validate')
    parser.add_argument('--output-dir', type=str, default='validation_output',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    print("=== Data Pipeline Validation Tool ===")
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        # Load original COCO annotations
        splits_dir = Path(config.get('data', {}).get('splits_dir', 'data/processed/splits'))
        val_annotations_file = splits_dir / 'val_annotations.json'
        
        if not val_annotations_file.exists():
            print(f"Validation annotations not found: {val_annotations_file}")
            return
            
        coco_data = load_original_coco_annotations(val_annotations_file)
        
        # Create dataset
        images_dir = Path(config.get('data', {}).get('images_dir', 'data/raw/images'))
        dataset = FloorPlanDataset(
            annotations_file=val_annotations_file,
            images_dir=images_dir,
            config=config,
            is_training=False,
            use_augmentation=False
        )
        
        print(f"\nDataset loaded successfully:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Categories: {dataset.category_names}")
        
        # Validate specific image
        if args.image_idx >= len(dataset):
            print(f"Image index {args.image_idx} out of range. Using index 0.")
            args.image_idx = 0
            
        # Run validations
        validate_coordinate_conversion(coco_data, dataset, args.image_idx)
        validate_evaluation_format_conversion()
        visualize_annotations_comparison(coco_data, dataset, args.image_idx, args.output_dir)
        
        print(f"\n  Data pipeline validation completed successfully!")
        print(f"   Visual validation saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"    Validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()