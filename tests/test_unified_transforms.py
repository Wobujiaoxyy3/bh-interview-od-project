#!/usr/bin/env python3
"""
Quick test of unified transform pipeline

This script validates that the new unified transforms work correctly
without causing import errors or data format issues.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_unified_transforms():
    """Test the unified transform pipeline"""
    print("Testing Unified Transform Pipeline")
    
    try:
        # Import unified transforms
        from datasets.unified_transforms import (
            create_unified_transform, 
            apply_unified_transform,
            validate_transform_output
        )
        print("SUCCESS: Successfully imported unified transforms")
        
        # Load configuration
        from config import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.load_config('faster_rcnn_config')
        print("SUCCESS: Successfully loaded configuration")
        
        # Create transform pipelines
        train_transform = create_unified_transform(config, is_training=True)
        val_transform = create_unified_transform(config, is_training=False)
        print("SUCCESS: Successfully created transform pipelines")
        
        # Create test data
        test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        test_boxes = [[100.0, 150.0, 300.0, 250.0], [200.0, 100.0, 400.0, 300.0]]
        test_labels = [1, 2]
        print("SUCCESS: Created test data")
        
        # Test validation transform
        print("\n--- Testing Validation Transform ---")
        val_result = apply_unified_transform(
            transform_pipeline=val_transform,
            image=test_image,
            bboxes=test_boxes,
            category_ids=test_labels
        )
        
        if validate_transform_output(val_result):
            print("PASS: Validation transform output is valid")
            print(f"   Image shape: {val_result['image'].shape}")
            print(f"   Bboxes count: {len(val_result.get('bboxes', []))}")
        else:
            print("FAIL: Validation transform output is invalid")
            return False
        
        # Test training transform
        print("\n--- Testing Training Transform ---")
        train_result = apply_unified_transform(
            transform_pipeline=train_transform,
            image=test_image,
            bboxes=test_boxes,
            category_ids=test_labels
        )
        
        if validate_transform_output(train_result):
            print("PASS: Training transform output is valid")
            print(f"   Image shape: {train_result['image'].shape}")
            print(f"   Bboxes count: {len(train_result.get('bboxes', []))}")
        else:
            print("FAIL: Training transform output is invalid")
            return False
        
        # Test empty bboxes case
        print("\n--- Testing Empty Bboxes Case ---")
        empty_result = apply_unified_transform(
            transform_pipeline=val_transform,
            image=test_image,
            bboxes=[],
            category_ids=[]
        )
        
        if validate_transform_output(empty_result):
            print("PASS: Empty bboxes case handled correctly")
        else:
            print("FAIL: Empty bboxes case failed")
            return False
        
        return True
        
    except ImportError as e:
        print(f"FAIL: Import error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_integration():
    """Test integration with dataset"""
    print("\nTesting Dataset Integration")
    
    try:
        # Test dataset loading with unified transforms
        from datasets.coco_dataset import FloorPlanDataset
        from config import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.load_config('faster_rcnn_config')
        
        data_config = config.get('data', {})
        images_dir = Path(data_config.get('images_dir', 'data/raw/images'))
        splits_dir = Path(data_config.get('splits_dir', 'data/processed/splits'))
        val_annotations = splits_dir / 'val_annotations.json'
        
        if not val_annotations.exists():
            print("WARNING: Validation annotations not found, skipping dataset integration test")
            return True
        
        # Create dataset with unified transforms
        val_dataset = FloorPlanDataset(
            annotations_file=val_annotations,
            images_dir=images_dir,
            config=config,
            is_training=False,
            use_augmentation=False
        )
        print("PASS: Successfully created dataset with unified transforms")
        
        if len(val_dataset) > 0:
            # Test loading first sample
            sample = val_dataset[0]
            
            # Check sample structure
            required_keys = ['image', 'target', 'image_id', 'image_info', 'original_size']
            for key in required_keys:
                if key not in sample:
                    print(f"FAIL: Missing sample key: {key}")
                    return False
            
            print("PASS: Sample has all required keys")
            
            # Check image tensor
            image = sample['image']
            if hasattr(image, 'shape') and len(image.shape) == 3:
                print(f"PASS: Image tensor shape correct: {image.shape}")
            else:
                print(f"FAIL: Invalid image tensor: {type(image)}")
                return False
            
            # Check target structure
            target = sample['target']
            required_target_keys = ['boxes', 'labels', 'area', 'iscrowd', 'image_id']
            for key in required_target_keys:
                if key not in target:
                    print(f"FAIL: Missing target key: {key}")
                    return False
            
            print("PASS: Target has all required keys")
            print(f"   Boxes shape: {target['boxes'].shape}")
            print(f"   Labels shape: {target['labels'].shape}")
            
            return True
        else:
            print("WARNING: Dataset is empty, cannot test sample loading")
            return True
            
    except Exception as e:
        print(f"FAIL: Dataset integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("Unified Transform Pipeline Test Suite")
    
    # Run tests
    test1_passed = test_unified_transforms()
    test2_passed = test_dataset_integration()
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    tests_passed = sum([test1_passed, test2_passed])
    total_tests = 2
    
    if tests_passed == total_tests:
        print(f"SUCCESS: ALL {total_tests} TESTS PASSED!")
        print("\nUnified transform pipeline is ready for training")
        return 0
    else:
        print(f"FAIL: {total_tests - tests_passed} out of {total_tests} tests failed")
        print("\nPlease fix issues before proceeding with training")
        return 1


if __name__ == '__main__':
    exit(main())