#!/usr/bin/env python3
"""
Debug MobileNet backbone structure to fix freezing issue
"""

import sys
from pathlib import Path
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def analyze_mobilenet_structure():
    """Analyze the actual MobileNet backbone structure"""
    
    print("=" * 80)
    print("DEBUGGING MOBILENET BACKBONE STRUCTURE")
    print("=" * 80)
    
    try:
        # Create MobileNet Faster R-CNN model
        print("1. Creating MobileNet Faster R-CNN model...")
        model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        
        print("\n2. OVERALL MODEL STRUCTURE")
        print("-" * 40)
        print("Model components:")
        for name, module in model.named_children():
            print(f"  - {name}: {type(module).__name__}")
        
        print("\n3. BACKBONE STRUCTURE")
        print("-" * 40)
        
        backbone = model.backbone
        print(f"Backbone type: {type(backbone).__name__}")
        print("Backbone components:")
        for name, module in backbone.named_children():
            print(f"  - {name}: {type(module).__name__}")
        
        print("\n4. BACKBONE.BODY STRUCTURE (The Problem Area)")
        print("-" * 40)
        
        backbone_body = backbone.body
        print(f"Backbone.body type: {type(backbone_body).__name__}")
        print("Backbone.body attributes:")
        
        # Check all attributes
        for attr_name in dir(backbone_body):
            if not attr_name.startswith('_'):
                attr = getattr(backbone_body, attr_name)
                if hasattr(attr, 'parameters'):  # It's a module
                    param_count = sum(p.numel() for p in attr.parameters())
                    print(f"  - {attr_name}: {type(attr).__name__} ({param_count:,} params)")
        
        print("\n5. DETAILED STRUCTURE INSPECTION")
        print("-" * 40)
        
        # Method 1: Check for 'features' attribute
        print("Method 1: Direct features access")
        if hasattr(backbone_body, 'features'):
            features = backbone_body.features
            print(f"  PASS: backbone_body.features exists: {type(features).__name__}")
            print(f"  PASS: Features length: {len(features)}")
            print(f"  PASS: Feature layers (first 5):")
            for i, layer in enumerate(features[:5]):
                print(f"    [{i}]: {type(layer).__name__}")
        else:
            print("  FAIL: backbone_body.features does NOT exist")
        
        # Method 2: Check for nested model
        print("\nMethod 2: Nested model access")
        if hasattr(backbone_body, 'model'):
            nested_model = backbone_body.model
            print(f"  PASS: backbone_body.model exists: {type(nested_model).__name__}")
            if hasattr(nested_model, 'features'):
                features = nested_model.features
                print(f"  PASS: backbone_body.model.features exists: {type(features).__name__}")
                print(f"  PASS: Features length: {len(features)}")
            else:
                print("  FAIL: backbone_body.model.features does NOT exist")
        else:
            print("  FAIL: backbone_body.model does NOT exist")
        
        # Method 3: Check IntermediateLayerGetter structure
        print("\nMethod 3: IntermediateLayerGetter inspection")
        if hasattr(backbone_body, 'return_layers'):
            print(f"  PASS: return_layers: {backbone_body.return_layers}")
        
        # Check if it's a Sequential or ModuleDict
        if hasattr(backbone_body, '_modules'):
            print(f"  PASS: _modules keys: {list(backbone_body._modules.keys())}")
            
            # Try to access through module keys
            for key, module in backbone_body._modules.items():
                if hasattr(module, 'features'):
                    print(f"  PASS: Found features in module '{key}': {type(module).__name__}")
                    features = module.features
                    print(f"    Features length: {len(features)}")
                    break
        
        print("\n6. TRYING DIFFERENT ACCESS PATTERNS")
        print("-" * 40)
        
        # Pattern 1: Direct iteration
        print("Pattern 1: Direct named_children iteration")
        for name, child in backbone_body.named_children():
            print(f"  {name}: {type(child).__name__}")
            if hasattr(child, 'features'):
                print(f"    ↳ Has 'features' attribute: {len(child.features)} layers")
        
        # Pattern 2: Check all children recursively
        print("\nPattern 2: Finding MobileNet features recursively")
        def find_mobilenet_features(module, path=""):
            if hasattr(module, 'features') and hasattr(module.features, '__len__'):
                return path, module.features
            
            for name, child in module.named_children():
                result = find_mobilenet_features(child, f"{path}.{name}" if path else name)
                if result[0]:
                    return result
            return None, None
        
        features_path, features = find_mobilenet_features(backbone_body)
        if features is not None:
            print(f"  PASS: Found MobileNet features at: {features_path}")
            print(f"  PASS: Features type: {type(features).__name__}")
            print(f"  PASS: Features length: {len(features)}")
            print(f"  PASS: Last 3 layers:")
            for i in range(max(0, len(features)-3), len(features)):
                print(f"    [{i}]: {type(features[i]).__name__}")
        else:
            print("  FAIL: Could not find MobileNet features")
        
        return features_path, features
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_freezing_fix(features_path, features):
    """Test the proposed fix for MobileNet freezing"""
    
    print("\n" + "=" * 80)
    print("TESTING MOBILENET FREEZING FIX")
    print("=" * 80)
    
    if features is None:
        print("Cannot test - no features found")
        return
    
    print(f"Using features at path: {features_path}")
    print(f"Total feature layers: {len(features)}")
    
    # Test different trainable_backbone_layers values
    for trainable_layers in [0, 1, 3, 5]:
        print(f"\n--- Testing trainable_backbone_layers = {trainable_layers} ---")
        
        # First freeze everything
        for param in features.parameters():
            param.requires_grad = False
        
        # Unfreeze the last N feature layers
        total_layers = len(features)
        start_idx = max(0, total_layers - trainable_layers)
        
        unfrozen_count = 0
        for i in range(start_idx, total_layers):
            for param in features[i].parameters():
                param.requires_grad = True
                unfrozen_count += param.numel()
        
        total_params = sum(p.numel() for p in features.parameters())
        print(f"  Unfroze layers {start_idx} to {total_layers-1}")
        print(f"  Unfrozen parameters: {unfrozen_count:,} / {total_params:,} "
              f"({100*unfrozen_count/total_params:.1f}%)")


def generate_fixed_code():
    """Generate the corrected MobileNet freezing code"""
    
    print("\n" + "=" * 80)
    print("PROPOSED FIX FOR MOBILENET FREEZING")
    print("=" * 80)
    
    fixed_code = '''
def _freeze_mobilenet_backbone(self, backbone_body):
    """Freeze MobileNet backbone layers - FIXED VERSION"""
    
    # First freeze everything
    for param in backbone_body.parameters():
        param.requires_grad = False
    
    # Find MobileNet features through various possible paths
    features = None
    features_path = ""
    
    # Method 1: Direct features access
    if hasattr(backbone_body, 'features'):
        features = backbone_body.features
        features_path = "backbone_body.features"
        
    # Method 2: Through wrapped model (IntermediateLayerGetter)  
    elif hasattr(backbone_body, 'model') and hasattr(backbone_body.model, 'features'):
        features = backbone_body.model.features
        features_path = "backbone_body.model.features"
        
    # Method 3: Search through all children recursively
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
        
        features, features_path = find_features(backbone_body)
    
    if features is not None:
        # Unfreeze the last N feature layers
        total_layers = len(features)
        start_idx = max(0, total_layers - self.trainable_backbone_layers)
        
        for i in range(start_idx, total_layers):
            for param in features[i].parameters():
                param.requires_grad = True
        
        logger.info(f"Unfroze MobileNet layers {start_idx} to {total_layers-1} from {features_path}")
    else:
        # Fallback: unfreeze all if we can't identify structure
        for param in backbone_body.parameters():
            param.requires_grad = True
        logger.warning("Could not identify MobileNet structure, unfroze all parameters")
    '''
    
    print(fixed_code)


def main():
    """Main debugging function"""
    
    print("MOBILENET BACKBONE STRUCTURE DEBUG")
    print("This script will help identify why MobileNet freezing fails")
    
    features_path, features = analyze_mobilenet_structure()
    
    if features is not None:
        test_freezing_fix(features_path, features)
    
    generate_fixed_code()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The MobileNet freezing issue is likely due to the backbone structure being wrapped
in an IntermediateLayerGetter, which changes how we access the features.

The fix involves:
1. Searching recursively for the MobileNet features
2. Using the found features path for parameter freezing
3. Adding better error handling and logging

Run this script to see the exact structure of your MobileNet model.
    """)


if __name__ == '__main__':
    main()