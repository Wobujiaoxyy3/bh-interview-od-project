#!/usr/bin/env python3
"""
Test script to verify coordinate transformation and inverse transformation
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from datasets import create_data_loaders
from config import ConfigManager

def test_coordinate_transform():
    """Test coordinate transformation logic"""
    
    print("=" * 60)
    print("Testing Coordinate Transformation")
    print("=" * 60)
    
    # Load config
    config_manager = ConfigManager()
    config = config_manager.load_config("faster_rcnn_config")
    
    # Create data loaders
    _, val_loader, _ = create_data_loaders(config)
    
    # Test a few samples
    for batch_idx, (images, targets) in enumerate(val_loader):
        if batch_idx >= 2:  # Test only 2 batches
            break
            
        print(f"\nBatch {batch_idx}:")
        for i, (image, target) in enumerate(zip(images, targets)):
            print(f"  Sample {i}:")
            print(f"    Image shape: {image.shape}")
            print(f"    Image ID: {target['image_id'].item()}")
            
            # Check transformation parameters
            if 'scale' in target:
                scale = target['scale'].item()
                pad_left = target['pad_left'].item()
                pad_top = target['pad_top'].item()
                original_size = target['original_size'].tolist()
                
                print(f"    Original size (H,W): {original_size}")
                print(f"    Scale factor: {scale:.4f}")
                print(f"    Padding (left, top): ({pad_left:.1f}, {pad_top:.1f})")
                
                # Verify transformation
                original_h, original_w = original_size
                target_size = 800
                
                expected_scale = min(target_size / original_w, target_size / original_h)
                new_width = int(original_w * expected_scale)
                new_height = int(original_h * expected_scale)
                expected_pad_left = (target_size - new_width) // 2
                expected_pad_top = (target_size - new_height) // 2
                
                print(f"    Expected scale: {expected_scale:.4f}")
                print(f"    Expected padding: ({expected_pad_left}, {expected_pad_top})")
                
                # Check if values match
                if abs(scale - expected_scale) < 0.001:
                    print("    PASS: Scale matches!")
                else:
                    print(f"    FAIL: Scale mismatch: {scale} vs {expected_scale}")
                    
                if abs(pad_left - expected_pad_left) < 1 and abs(pad_top - expected_pad_top) < 1:
                    print("    PASS: Padding matches!")
                else:
                    print(f"    FAIL: Padding mismatch")
                    
                # Test coordinate transformation
                if len(target['boxes']) > 0:
                    box = target['boxes'][0]  # First box in transformed coordinates
                    x1, y1, x2, y2 = box.tolist()
                    
                    print(f"    Transformed box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                    
                    # Apply inverse transform
                    x1_orig = (x1 - pad_left) / scale
                    y1_orig = (y1 - pad_top) / scale
                    x2_orig = (x2 - pad_left) / scale
                    y2_orig = (y2 - pad_top) / scale
                    
                    print(f"    Original box (computed): [{x1_orig:.1f}, {y1_orig:.1f}, {x2_orig:.1f}, {y2_orig:.1f}]")
                    
                    # Check if coordinates are within original image bounds
                    if (0 <= x1_orig <= original_w and 0 <= x2_orig <= original_w and
                        0 <= y1_orig <= original_h and 0 <= y2_orig <= original_h):
                        print("    PASS: Inverse transform produces valid coordinates!")
                    else:
                        print("    FAIL: Inverse transform produces out-of-bounds coordinates!")
            else:
                print("    WARNING: No transformation parameters found in target!")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == '__main__':
    test_coordinate_transform()