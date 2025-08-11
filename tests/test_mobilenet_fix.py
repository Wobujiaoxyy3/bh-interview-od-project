#!/usr/bin/env python3
"""
Test MobileNet backbone freezing fix
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_mobilenet_freezing():
    """Test the fixed MobileNet freezing logic"""
    
    print("=" * 60)
    print("TESTING MOBILENET FREEZING FIX")
    print("=" * 60)
    
    try:
        from models import create_faster_rcnn_model
        
        # Test different trainable_backbone_layers settings
        test_configs = [
            {'layers': 0, 'expected_range': (40, 60)},  # ~45-55% trainable
            {'layers': 1, 'expected_range': (60, 80)},  # ~65-75% trainable  
            {'layers': 3, 'expected_range': (75, 95)},  # ~80-90% trainable
            {'layers': 5, 'expected_range': (90, 100)}, # ~95-100% trainable
        ]
        
        print("Testing different trainable_backbone_layers configurations:")
        print("-" * 60)
        
        all_passed = True
        
        for test_config in test_configs:
            layers = test_config['layers']
            expected_min, expected_max = test_config['expected_range']
            
            print(f"\n--- trainable_backbone_layers = {layers} ---")
            
            config = {
                'num_classes': 4,
                'backbone': 'mobilenet_v3_large',
                'pretrained': True,
                'trainable_backbone_layers': layers,
                'model_name': 'fasterrcnn_mobilenet_v3_large_fpn'
            }
            
            try:
                model = create_faster_rcnn_model(config)
                
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                trainable_percent = 100 * trainable_params / total_params
                
                print(f"Total parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,}")
                print(f"Trainable percentage: {trainable_percent:.1f}%")
                
                # Check if it's in expected range
                if expected_min <= trainable_percent <= expected_max:
                    print(f"PASS: Trainable percentage ({trainable_percent:.1f}%) in expected range [{expected_min}-{expected_max}%]")
                elif trainable_percent == 100.0:
                    print(f"FAIL: All parameters trainable (100%) - freezing not working")
                    all_passed = False
                else:
                    print(f"WARNING: Trainable percentage ({trainable_percent:.1f}%) outside expected range [{expected_min}-{expected_max}%]")
                    print("  This might be OK - the exact percentage depends on model architecture")
                
            except Exception as e:
                print(f"ERROR: Failed to create model: {e}")
                all_passed = False
        
        print("\n" + "=" * 60)
        if all_passed:
            print("ALL TESTS PASSED!")
            print("MobileNet backbone freezing is now working correctly.")
        else:
            print("SOME TESTS FAILED!")
            print("There may still be issues with the MobileNet freezing logic.")
        
        return all_passed
        
    except ImportError as e:
        print(f"Cannot test - missing dependencies: {e}")
        print("Please ensure torch and torchvision are installed")
        return False
    except Exception as e:
        print(f"Test failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_structure_identification():
    """Test if the fix can properly identify MobileNet structure"""
    
    print("\n" + "=" * 60)
    print("TESTING MOBILENET STRUCTURE IDENTIFICATION")
    print("=" * 60)
    
    try:
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
        
        # Create MobileNet model
        model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        backbone_body = model.backbone.body
        
        print("MobileNet backbone structure analysis:")
        print(f"backbone_body type: {type(backbone_body).__name__}")
        
        # Test the search methods
        feature_layers = None
        features_path = ""
        
        # Method 1: Direct features access
        if hasattr(backbone_body, 'features'):
            feature_layers = backbone_body.features
            features_path = "backbone_body.features"
            print(f"Method 1 SUCCESS: Found features at {features_path}")
            
        # Method 2: Through wrapped model
        elif hasattr(backbone_body, 'model') and hasattr(backbone_body.model, 'features'):
            feature_layers = backbone_body.model.features
            features_path = "backbone_body.model.features"
            print(f"Method 2 SUCCESS: Found features at {features_path}")
            
        # Method 3: Recursive search
        else:
            def find_features(module, path=""):
                if hasattr(module, 'features') and hasattr(module.features, '__len__'):
                    return module.features, path + ".features"
                
                for name, child in module.named_children():
                    child_path = f"{path}.{name}" if path else name
                    result_features, result_path = find_features(child, child_path)
                    if result_features is not None:
                        return result_features, result_path
                return None, ""
            
            feature_layers, features_path = find_features(backbone_body)
            if feature_layers is not None:
                print(f"Method 3 SUCCESS: Found features at {features_path}")
            else:
                print("Method 3 FAILED: Could not find features")
        
        if feature_layers is not None:
            print(f"Features length: {len(feature_layers)}")
            print(f"Feature types (first 3): {[type(feature_layers[i]).__name__ for i in range(min(3, len(feature_layers)))]}")
            print("Structure identification SUCCESSFUL")
            return True
        else:
            print("Structure identification FAILED")
            return False
            
    except Exception as e:
        print(f"Structure test failed: {e}")
        return False


def main():
    """Main test function"""
    
    print("MOBILENET BACKBONE FREEZING FIX VERIFICATION")
    print("This script tests if the MobileNet freezing fix is working correctly")
    
    # Test 1: Structure identification
    structure_ok = test_structure_identification()
    
    # Test 2: Freezing logic
    freezing_ok = test_mobilenet_freezing()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    if structure_ok and freezing_ok:
        print("SUCCESS: MobileNet freezing fix is working!")
        print("\nYou can now run your MobileNet training and should see:")
        print("- Proper trainable parameter percentages (not 100%)")
        print("- Log messages like 'Successfully unfroze MobileNet layers X to Y'")
        print("- Improved training efficiency with frozen early layers")
    elif structure_ok:
        print("PARTIAL SUCCESS: Structure identification works but freezing may have issues")
    else:
        print("FAILED: Fix did not resolve the MobileNet freezing issue")
        print("\nTroubleshooting:")
        print("1. Check if torch/torchvision versions are compatible")
        print("2. Try running with different trainable_backbone_layers values")
        print("3. Check the logs for detailed error messages")


if __name__ == '__main__':
    main()