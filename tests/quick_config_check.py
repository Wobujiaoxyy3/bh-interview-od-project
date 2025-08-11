"""
Quick configuration and data consistency check
Updated to work with current base_config.yaml format
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ConfigManager


def main():
    print("Floor Plan Object Detection - Configuration Check")
    print("=" * 60)
    
    # Load config
    config_manager = ConfigManager()
    config = config_manager.load_config('faster_rcnn_config')
    
    # Display project information
    project_config = config.get('project', {})
    print(f"\nProject Information:")
    print(f"  Name: {project_config.get('name', 'Unknown')}")
    print(f"  Description: {project_config.get('description', 'Unknown')}")
    print(f"  Version: {project_config.get('version', 'Unknown')}")
    print(f"  Author: {project_config.get('author', 'Unknown')}")
    
    # Check model config
    model_config = config.get('model', {})
    num_classes = model_config.get('num_classes', 0)
    architecture = model_config.get('architecture', 'unknown')
    backbone = model_config.get('backbone', 'unknown')
    pretrained = model_config.get('pretrained', False)
    
    print(f"\nModel Configuration:")
    print(f"  Architecture: {architecture}")
    print(f"  Backbone: {backbone}")
    print(f"  Pretrained: {pretrained}")
    print(f"  Number of classes: {num_classes}")
    
    # Check data config
    data_config = config.get('data', {})
    annotations_file = Path(data_config.get('annotations_file', ''))
    images_dir = Path(data_config.get('images_dir', ''))
    
    print(f"\nData Configuration:")
    print(f"  Images directory: {images_dir}")
    print(f"  Annotations file: {annotations_file}")
    
    # Load annotations to check categories
    if annotations_file.exists():
        try:
            with open(annotations_file, 'r') as f:
                coco_data = json.load(f)
                
            categories = coco_data.get('categories', [])
            actual_categories = len(categories)
            
            print(f"  Actual categories in data: {actual_categories}")
            print(f"  Categories:")
            for cat in categories:
                print(f"    ID {cat['id']}: {cat['name']}")
                
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"  ERROR: Could not load annotations file: {e}")
            actual_categories = 0
            categories = []
            
    else:
        print(f"  ERROR: Annotations file not found: {annotations_file}")
        actual_categories = 0
        categories = []
    
    # Check preprocessing config
    preprocessing_config = config.get('preprocessing', {})
    print(f"\nPreprocessing Configuration:")
    print(f"  Target size: {preprocessing_config.get('target_size', 'unknown')}")
    print(f"  CLAHE enabled: {preprocessing_config.get('apply_clahe', False)}")
    print(f"  Normalization mean: {preprocessing_config.get('normalize_mean', 'unknown')}")
    print(f"  Normalization std: {preprocessing_config.get('normalize_std', 'unknown')}")
    
    # Check training config
    training_config = config.get('training', {})
    optimizer_config = config.get('optimizer', {})
    data_loader_config = config.get('data_loader', {})
    
    print(f"\nTraining Configuration:")
    print(f"  Learning rate: {optimizer_config.get('learning_rate', 'unknown')}")
    print(f"  Optimizer type: {optimizer_config.get('type', 'unknown')}")
    print(f"  Weight decay: {optimizer_config.get('weight_decay', 'unknown')}")
    print(f"  Batch size: {data_loader_config.get('batch_size', 'unknown')}")
    print(f"  Num workers: {data_loader_config.get('num_workers', 'unknown')}")
    print(f"  Epochs: {training_config.get('num_epochs', 'unknown')}")
    print(f"  Gradient clipping: {training_config.get('gradient_clipping', 'unknown')}")
    
    # Check scheduler config
    scheduler_config = config.get('scheduler', {})
    print(f"\nScheduler Configuration:")
    print(f"  Use scheduler: {scheduler_config.get('use_scheduler', False)}")
    print(f"  Type: {scheduler_config.get('type', 'unknown')}")
    if scheduler_config.get('type') == 'reduce_on_plateau':
        print(f"  Mode: {scheduler_config.get('mode', 'unknown')}")
        print(f"  Factor: {scheduler_config.get('factor', 'unknown')}")
        print(f"  Patience: {scheduler_config.get('patience', 'unknown')}")
    
    # Check evaluation config
    evaluation_config = config.get('evaluation', {})
    print(f"\nEvaluation Configuration:")
    print(f"  Confidence threshold: {evaluation_config.get('confidence_threshold', 'unknown')}")
    if 'iou_thresholds' in evaluation_config:
        print(f"  IoU thresholds: {evaluation_config.get('iou_thresholds', 'unknown')}")
    if 'nms_threshold' in evaluation_config:
        print(f"  NMS threshold: {evaluation_config.get('nms_threshold', 'unknown')}")
    
    # Consistency checks
    print(f"\n" + "=" * 60)
    print("Consistency Checks:")
    
    # Check class consistency
    if actual_categories > 0:
        # For object detection, num_classes = actual_categories + 1 (background)
        expected_num_classes = actual_categories + 1
        if num_classes == expected_num_classes:
            print(f"  ✓ PASS: num_classes ({num_classes}) = categories ({actual_categories}) + background (1)")
        else:
            print(f"  ✗ FAIL: num_classes ({num_classes}) != categories ({actual_categories}) + background (1) = {expected_num_classes}")
            print(f"       Please update config: num_classes should be {expected_num_classes}")
    else:
        print(f"  ⚠  WARNING: Could not verify class consistency - no categories loaded")
    
    # Check pretrained setting
    if pretrained:
        print(f"  ✓ PASS: Pretrained weights enabled")
    else:
        print(f"  ⚠  WARNING: Pretrained weights disabled - may slow training convergence")
    
    # Check data paths
    if images_dir.exists():
        print(f"  ✓ PASS: Images directory exists: {images_dir}")
    else:
        print(f"  ✗ FAIL: Images directory not found: {images_dir}")
    
    if annotations_file.exists():
        print(f"  ✓ PASS: Annotations file exists: {annotations_file}")
    else:
        print(f"  ✗ FAIL: Annotations file not found: {annotations_file}")
    
    # Check device configuration
    device = config.get('device', 'unknown')
    print(f"  Device: {device}")
    
    # Check output directory configuration
    output_dir = config.get('output_dir', 'unknown')
    print(f"  Output directory: {output_dir}")
    
    print(f"\n" + "=" * 60)
    print("Configuration check completed.")


if __name__ == '__main__':
    main()