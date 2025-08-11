"""
Visual validation script for coordinate transformations
Shows before/after images with bounding boxes to verify correctness
Follows the exact data pipeline: augmentation -> transform
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
import sys

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def create_test_image_with_objects():
    """Create a test image with clear visual objects and corresponding bboxes"""
    # Create 400x600 image (height=400, width=600)
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255  # White background
    
    # Draw distinct colored rectangles as "objects"
    # Object 1: Red rectangle (door)
    cv2.rectangle(image, (50, 50), (150, 200), (0, 0, 255), -1)
    
    # Object 2: Green rectangle (window)  
    cv2.rectangle(image, (400, 100), (550, 150), (0, 255, 0), -1)
    
    # Object 3: Blue rectangle (room)
    cv2.rectangle(image, (200, 250), (450, 350), (255, 0, 0), -1)
    
    # Add some text for orientation reference
    cv2.putText(image, "TOP", (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(image, "LEFT", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Define corresponding bboxes [x1, y1, x2, y2] 
    bboxes = [
        [50, 50, 150, 200],    # Red door
        [400, 100, 550, 150],  # Green window
        [200, 250, 450, 350]   # Blue room
    ]
    
    labels = [1, 2, 3]  # door, window, room
    class_names = {1: 'door', 2: 'window', 3: 'room'}
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR for opencv
    
    return image, bboxes, labels, class_names, colors

def visualize_image_with_bboxes(image, bboxes, labels, class_names, colors, title):
    """Visualize image with bounding boxes"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Convert BGR to RGB for matplotlib
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    ax.imshow(image_rgb)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Draw bounding boxes
    for bbox, label, color in zip(bboxes, labels, colors):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Convert BGR to RGB for matplotlib
        color_rgb = (color[2]/255, color[1]/255, color[0]/255)
        
        # Create rectangle
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor=color_rgb, facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        ax.text(x1, y1-10, f'{class_names[label]}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color_rgb, alpha=0.7),
                fontsize=10, fontweight='bold')
        
        # Add coordinates text
        ax.text(x1+5, y1+15, f'({x1},{y1})\n({x2},{y2})', 
                fontsize=8, color='white', fontweight='bold')
    
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)  # Invert y-axis
    ax.axis('on')
    ax.grid(True, alpha=0.3)
    
    return fig, ax

def test_flip_augmentation():
    """Test flip augmentation with visualization"""
    print("Testing Flip Augmentation")
    
    # Import here to avoid dependency issues
    try:
        from src.datasets.augmentations import FlipAugmentation
    except ImportError:
        # Simple implementation for testing
        class FlipAugmentation:
            def __init__(self, horizontal_prob=0.5, vertical_prob=0.0):
                self.horizontal_prob = horizontal_prob
                self.vertical_prob = vertical_prob
            
            def __call__(self, image, bboxes=None):
                import random
                height, width = image.shape[:2]
                flip_horizontal = random.random() < self.horizontal_prob
                flip_vertical = random.random() < self.vertical_prob
                
                # Apply flips to image
                if flip_horizontal:
                    image = cv2.flip(image, 1)
                if flip_vertical:
                    image = cv2.flip(image, 0)
                
                # Transform bounding boxes (using FIXED logic)
                if bboxes is not None:
                    bboxes = np.array(bboxes)
                    if flip_horizontal:
                        x1_orig = bboxes[:, 0].copy()
                        x2_orig = bboxes[:, 2].copy()
                        bboxes[:, 0] = width - x2_orig
                        bboxes[:, 2] = width - x1_orig
                    
                    if flip_vertical:
                        y1_orig = bboxes[:, 1].copy()
                        y2_orig = bboxes[:, 3].copy()
                        bboxes[:, 1] = height - y2_orig
                        bboxes[:, 3] = height - y1_orig
                    
                    bboxes = bboxes.tolist()
                
                return image, bboxes, {'flip_horizontal': flip_horizontal, 'flip_vertical': flip_vertical}
    
    # Create test data
    image, bboxes, labels, class_names, colors = create_test_image_with_objects()
    
    # Test horizontal flip
    flip_aug = FlipAugmentation(horizontal_prob=1.0, vertical_prob=0.0)
    h_flipped_image, h_flipped_bboxes, h_info = flip_aug(image.copy(), bboxes.copy())
    
    # Test vertical flip  
    v_flip_aug = FlipAugmentation(horizontal_prob=0.0, vertical_prob=1.0)
    v_flipped_image, v_flipped_bboxes, v_info = v_flip_aug(image.copy(), bboxes.copy())
    
    # Test both flips
    both_flip_aug = FlipAugmentation(horizontal_prob=1.0, vertical_prob=1.0)
    both_flipped_image, both_flipped_bboxes, both_info = both_flip_aug(image.copy(), bboxes.copy())
    
    # Create visualization
    fig = plt.figure(figsize=(20, 15))
    
    # Original
    plt.subplot(2, 2, 1)
    fig1, ax1 = visualize_image_with_bboxes(image, bboxes, labels, class_names, colors, 
                                           "Original Image")
    
    # Horizontal flip
    plt.subplot(2, 2, 2)
    fig2, ax2 = visualize_image_with_bboxes(h_flipped_image, h_flipped_bboxes, labels, class_names, colors,
                                           "Horizontal Flip")
    
    # Vertical flip
    plt.subplot(2, 2, 3) 
    fig3, ax3 = visualize_image_with_bboxes(v_flipped_image, v_flipped_bboxes, labels, class_names, colors,
                                           "Vertical Flip")
    
    # Both flips
    plt.subplot(2, 2, 4)
    fig4, ax4 = visualize_image_with_bboxes(both_flipped_image, both_flipped_bboxes, labels, class_names, colors,
                                           "Both Flips")
    
    plt.tight_layout()
    plt.savefig('flip_augmentation_test.png', dpi=150, bbox_inches='tight')
    print("Flip augmentation visualization saved as 'flip_augmentation_test.png'")
    
    return image, bboxes, labels, class_names, colors

def test_rotation_augmentation(original_image, original_bboxes, labels, class_names, colors):
    """Test rotation augmentation with visualization"""
    print("Testing Rotation Augmentation")
    
    # Import or implement rotation
    try:
        from src.datasets.augmentations import Rotation90
    except ImportError:
        class Rotation90:
            def __init__(self, probability=0.5, angles=[0, 90, 180, 270]):
                self.probability = probability
                self.angles = angles
                
            def _rotate_image_90(self, image, angle):
                if angle == 0:
                    return image
                elif angle == 90:
                    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    return cv2.rotate(image, cv2.ROTATE_180)
                elif angle == 270:
                    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
            def _rotate_bbox_90(self, bbox, image_shape, angle):
                height, width = image_shape
                x1, y1, x2, y2 = bbox
                
                if angle == 0:
                    return bbox
                elif angle == 90:
                    # Fixed formula: (x, y) -> (y, height - x)
                    new_x1 = y1
                    new_y1 = height - x2
                    new_x2 = y2
                    new_y2 = height - x1
                elif angle == 180:
                    new_x1 = width - x2
                    new_y1 = height - y2
                    new_x2 = width - x1
                    new_y2 = height - y1
                elif angle == 270:
                    new_x1 = height - y2
                    new_y1 = x1
                    new_x2 = height - y1
                    new_y2 = x2
                
                return [min(new_x1, new_x2), min(new_y1, new_y2), 
                       max(new_x1, new_x2), max(new_y1, new_y2)]
            
            def __call__(self, image, bboxes=None):
                # For testing, always apply rotation
                import random
                angle = random.choice(self.angles)
                
                rotated_image = self._rotate_image_90(image, angle)
                
                if bboxes is not None:
                    original_shape = image.shape[:2]
                    rotated_bboxes = []
                    for bbox in bboxes:
                        rotated_bbox = self._rotate_bbox_90(bbox, original_shape, angle)
                        rotated_bboxes.append(rotated_bbox)
                else:
                    rotated_bboxes = None
                    
                return rotated_image, rotated_bboxes, {'angle': angle}
    
    # Test different rotations
    rotation_results = []
    for angle in [0, 90, 180, 270]:
        rot_aug = Rotation90(probability=1.0, angles=[angle])
        rotated_image, rotated_bboxes, info = rot_aug(original_image.copy(), original_bboxes.copy())
        rotation_results.append((rotated_image, rotated_bboxes, f"Rotation {angle}°"))
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, (rot_image, rot_bboxes, title) in enumerate(rotation_results):
        ax = axes[i]
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(rot_image, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Draw bounding boxes
        for bbox, label, color in zip(rot_bboxes, labels, colors):
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # Convert BGR to RGB for matplotlib
            color_rgb = (color[2]/255, color[1]/255, color[0]/255)
            
            # Create rectangle
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=2, edgecolor=color_rgb, facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            ax.text(x1, y1-10, f'{class_names[label]}', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color_rgb, alpha=0.7),
                    fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, rot_image.shape[1])
        ax.set_ylim(rot_image.shape[0], 0)
        ax.axis('on')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rotation_augmentation_test.png', dpi=150, bbox_inches='tight')
    print("Rotation augmentation visualization saved as 'rotation_augmentation_test.png'")

def test_full_pipeline():
    """Test the complete augmentation -> transform pipeline"""
    print("Testing Complete Data Pipeline (Augmentation -> Transform)")
    
    # Simulate the exact pipeline from coco_dataset.py
    image, bboxes, labels, class_names, colors = create_test_image_with_objects()
    
    # Step 1: Apply augmentation (flip + rotation)
    try:
        from src.datasets.augmentations import FlipAugmentation, Rotation90
        
        # Create augmentation pipeline (similar to FloorPlanAugmentation)
        flip_aug = FlipAugmentation(horizontal_prob=1.0, vertical_prob=0.0)
        rot_aug = Rotation90(probability=1.0, angles=[90])
        
        # Apply flip first
        aug1_image, aug1_bboxes, flip_info = flip_aug(image.copy(), bboxes.copy())
        
        # Then apply rotation
        aug2_image, aug2_bboxes, rot_info = rot_aug(aug1_image, aug1_bboxes)
        
        print(f"Applied: {flip_info}, {rot_info}")
        
    except ImportError:
        print("Using simplified augmentation for testing")
        aug2_image, aug2_bboxes = image.copy(), bboxes.copy()
    
    # Step 2: Apply transforms (resize with padding - simplified)
    # Simulate AlbumentationsTransform resize to 800x800
    target_size = 800
    h, w = aug2_image.shape[:2]
    
    # Calculate scale to fit within target size while maintaining aspect ratio
    scale = min(target_size/w, target_size/h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized_image = cv2.resize(aug2_image, (new_w, new_h))
    
    # Pad to target size
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    
    final_image = np.ones((target_size, target_size, 3), dtype=np.uint8) * 128  # Gray padding
    final_image[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_image
    
    # Transform bboxes
    final_bboxes = []
    for bbox in aug2_bboxes:
        x1, y1, x2, y2 = bbox
        # Apply scale and padding
        new_x1 = x1 * scale + pad_x
        new_y1 = y1 * scale + pad_y
        new_x2 = x2 * scale + pad_x
        new_y2 = y2 * scale + pad_y
        final_bboxes.append([new_x1, new_y1, new_x2, new_y2])
    
    # Visualize pipeline steps
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Step 1: Original
    ax1 = axes[0]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(image_rgb)
    ax1.set_title("1. Original Image", fontsize=14, fontweight='bold')
    
    for bbox, label, color in zip(bboxes, labels, colors):
        x1, y1, x2, y2 = bbox
        color_rgb = (color[2]/255, color[1]/255, color[0]/255)
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=color_rgb, facecolor='none')
        ax1.add_patch(rect)
        ax1.text(x1, y1-10, f'{class_names[label]}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color_rgb, alpha=0.7))
    
    ax1.axis('on')
    ax1.grid(True, alpha=0.3)
    
    # Step 2: After augmentation
    ax2 = axes[1]
    aug_image_rgb = cv2.cvtColor(aug2_image, cv2.COLOR_BGR2RGB)
    ax2.imshow(aug_image_rgb)
    ax2.set_title("2. After Augmentation", fontsize=14, fontweight='bold')
    
    for bbox, label, color in zip(aug2_bboxes, labels, colors):
        x1, y1, x2, y2 = bbox
        color_rgb = (color[2]/255, color[1]/255, color[0]/255)
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=color_rgb, facecolor='none')
        ax2.add_patch(rect)
        ax2.text(x1, y1-10, f'{class_names[label]}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color_rgb, alpha=0.7))
    
    ax2.axis('on')
    ax2.grid(True, alpha=0.3)
    
    # Step 3: Final (after transforms)
    ax3 = axes[2]
    final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    ax3.imshow(final_image_rgb)
    ax3.set_title("3. Final (Resized + Padded)", fontsize=14, fontweight='bold')
    
    for bbox, label, color in zip(final_bboxes, labels, colors):
        x1, y1, x2, y2 = bbox
        color_rgb = (color[2]/255, color[1]/255, color[0]/255)
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=color_rgb, facecolor='none')
        ax3.add_patch(rect)
        ax3.text(x1, y1-10, f'{class_names[label]}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color_rgb, alpha=0.7))
    
    ax3.axis('on')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('complete_pipeline_test.png', dpi=150, bbox_inches='tight')
    print(" Complete pipeline visualization saved as 'complete_pipeline_test.png'")

def main():
    """Run all visual validation tests"""
    print("Visual Coordinate Transform Validation")
    print("=" * 60)
    print("This script validates the fixed coordinate transformations with visualization")
    print("Order: Augmentation -> Transform (matching actual data pipeline)")
    print("=" * 60)
    
    # Test flip augmentation
    original_image, original_bboxes, labels, class_names, colors = test_flip_augmentation()
    
    print()
    
    # Test rotation augmentation
    test_rotation_augmentation(original_image, original_bboxes, labels, class_names, colors)
    
    print()
    
    # Test complete pipeline
    test_full_pipeline()
    
    print("\n" + "=" * 60)
    print(" Validation Complete!")
    print("Check the generated images:")
    print("  - flip_augmentation_test.png")
    print("  - rotation_augmentation_test.png") 
    print("  - complete_pipeline_test.png")
    print("=" * 60)
    
    print("\ What to verify:")
    print("1. Bounding boxes should correctly follow the objects after transformations")
    print("2. No boxes should be completely outside image boundaries")
    print("3. Box coordinates should be reasonable (positive, within image)")
    print("4. Visual alignment between objects and boxes should be perfect")

if __name__ == "__main__":
    main()