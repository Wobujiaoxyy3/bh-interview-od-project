#!/usr/bin/env python3
"""
Test script to verify framework imports work correctly
Updated to match current module structure
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_framework_imports():
    """Test all framework imports systematically"""
    
    print("Testing framework imports...")
    print("=" * 60)
    
    try:
        print("\n1. Testing config imports...")
        from config import ConfigManager, GridSearchManager, GridSearchRunner
        print("  ✓ ConfigManager imported successfully")
        print("  ✓ GridSearchManager imported successfully")
        print("  ✓ GridSearchRunner imported successfully")
        
        print("\n2. Testing dataset imports...")
        from datasets import (
            DataSplitter, split_dataset,
            FloorPlanDataset, create_datasets, create_data_loaders, collate_fn,
            create_unified_transform, create_training_transform, create_validation_transform,
            apply_unified_transform, validate_transform_output
        )
        print("  DataSplitter and split_dataset imported successfully")
        print("  FloorPlanDataset and related functions imported successfully")
        print("  Unified transform functions imported successfully")
        
        print("\n3. Testing evaluation imports...")
        from evaluation import COCOEvaluator, create_evaluator
        print("  COCOEvaluator imported successfully")
        print("  create_evaluator imported successfully")
        
        print("\n4. Testing utils imports...")
        from utils import setup_logging, set_random_seed
        print("  setup_logging imported successfully")
        print("  set_random_seed imported successfully")
        
        print("\n5. Testing custom types imports...")
        try:
            from custom_types import BoundingBox, ObjectDetectionResult, TrainingMetrics
            print("  Custom data types imported successfully")
        except ImportError as e:
            print(f"   Custom types import failed (optional): {e}")
        
        print("\n6. Testing model imports (requires torch)...")
        try:
            from models import (
                FloorPlanFasterRCNN, SimpleFasterRCNN, create_faster_rcnn_model, load_faster_rcnn_checkpoint,
                FloorPlanRetinaNet, SimpleRetinaNet, FocalLoss, create_retinanet_model, load_retinanet_checkpoint
            )
            print("  Faster R-CNN models imported successfully")
            print("  RetinaNet models imported successfully")
        except ImportError as e:
            print(f"   Model imports failed (requires torch/torchvision): {e}")
        
        print("\n7. Testing training imports (requires torch)...")
        try:
            from training import ObjectDetectionTrainer, train_model
            print("  Training modules imported successfully")
        except ImportError as e:
            print(f"  ⚠  Training imports failed (requires torch/torchvision): {e}")
        
        print("\n" + "=" * 60)
        print(" FRAMEWORK CORE IMPORTS SUCCESSFUL!")
        print("=" * 60)
        
        print("\nNote: Some imports require additional packages:")
        print("  - torch, torchvision for models and training")
        print("  - pycocotools for COCO evaluation") 
        print("  - opencv-python for image processing")
        print("\nInstall missing packages with:")
        print("  pip install torch torchvision pycocotools opencv-python")
        
        return True
        
    except ImportError as e:
        print(f"\n[ERROR] Import error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_functionality():
    """Test basic config functionality"""
    
    print("\n" + "=" * 60)
    print("Testing Configuration Functionality")
    print("=" * 60)
    
    try:
        from config import ConfigManager
        
        config_manager = ConfigManager()
        available_configs = config_manager.get_available_configs()
        
        print(f"Available configurations: {available_configs}")
        
        if 'faster_rcnn_config' in available_configs:
            print("\nTesting faster_rcnn_config loading...")
            config = config_manager.load_config('faster_rcnn_config')
            print("  ✓ Configuration loaded successfully")
            
            # Check key sections
            required_sections = ['project', 'model', 'data', 'training', 'evaluation']
            missing_sections = []
            
            for section in required_sections:
                if section in config:
                    print(f"  {section} section present")
                else:
                    missing_sections.append(section)
                    print(f"  {section} section missing")
            
            if not missing_sections:
                print("  All required configuration sections present")
                return True
            else:
                print(f"   Missing sections: {missing_sections}")
                return False
        else:
            print("    faster_rcnn_config not found")
            return False
            
    except Exception as e:
        print(f"   Configuration test failed: {e}")
        return False


def main():
    """Main test function"""
    
    success = True
    
    # Test framework imports
    success &= test_framework_imports()
    
    # Test config functionality
    success &= test_config_functionality()
    
    print("\n" + "=" * 60)
    if success:
        print(" ALL TESTS PASSED!")
        print("Framework is ready for use.")
    else:
        print(" SOME TESTS FAILED!")
        print("Please check the errors above and install missing dependencies.")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())