"""
YOLO to COCO Conversion Validation Script
Validates the accuracy and completeness of the YOLO to COCO format conversion

"""

import json
import os
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Tuple, Set
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict


class ConversionValidator:
    """
    Validator class for YOLO to COCO conversion verification
    """
    
    def __init__(self, 
                 original_images_dir: str,
                 original_annotations_dir: str, 
                 coco_annotations_file: str,
                 output_dir: str = "validation_results"):
        """
        Initialize the validator
        
        Args:
            original_images_dir: Path to original images
            original_annotations_dir: Path to original YOLO annotations
            coco_annotations_file: Path to converted COCO annotations
            output_dir: Directory to save validation results
        """
        self.images_dir = Path(original_images_dir)
        self.yolo_annotations_dir = Path(original_annotations_dir)
        self.coco_file = Path(coco_annotations_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load COCO annotations
        with open(self.coco_file, 'r') as f:
            self.coco_data = json.load(f)

        # Class mapping for validation
        self.yolo_to_coco_class = {0: 1, 1: 2, 2: 3}  # door, window, room
        self.class_names = {1: "door", 2: "window", 3: "room"}
        
        # Validation results
        self.validation_results = {
            "total_images_expected": 0,
            "total_images_converted": 0,
            "missing_images": [],
            "total_annotations_expected": 0,
            "total_annotations_converted": 0,
            "coordinate_validation": {
                "samples_tested": 0,
                "coordinate_errors": [],
                "bbox_out_of_bounds": []
            },
            "class_mapping_validation": {
                "correct_mappings": 0,
                "incorrect_mappings": []
            }
        }
    
    def get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """Get image dimensions"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        height, width = img.shape[:2]
        return width, height
    
    def yolo_to_coco_bbox_reference(self, yolo_bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """
        Reference implementation of YOLO to COCO bbox conversion for validation
        """
        center_x_norm, center_y_norm, width_norm, height_norm = yolo_bbox

        # Convert to absolute coordinates
        abs_center_x = center_x_norm * img_width
        abs_center_y = center_y_norm * img_height
        abs_width = width_norm * img_width
        abs_height = height_norm * img_height
        
        # Convert to top-left coordinates
        top_left_x = abs_center_x - abs_width / 2
        top_left_y = abs_center_y - abs_height / 2
        
        return [top_left_x, top_left_y, abs_width, abs_height]
    
    def validate_data_completeness(self) -> Dict:
        """
        Validate that all images and annotations were converted
        """
        print("Validating data completeness...")

        # Get all original image files
        image_extensions = ['.png', '.jpg', '.jpeg']
        original_images = set()
        for ext in image_extensions:
            original_images.update([f.name for f in self.images_dir.glob(f'*{ext}')])
        
        # Get converted image filenames
        converted_images = {img['file_name'] for img in self.coco_data['images']}
        
        # Check for missing images
        missing_images = original_images - converted_images
        
        # Count original annotations
        total_original_annotations = 0
        for image_name in original_images:
            annotation_file = image_name.replace('.png', '.txt').replace('.jpg', '.txt')
            annotation_path = self.yolo_annotations_dir / annotation_file
            
            if annotation_path.exists():
                with open(annotation_path, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    total_original_annotations += len(lines)

        # Update results
        self.validation_results.update({
            "total_images_expected": len(original_images),
            "total_images_converted": len(converted_images),
            "missing_images": list(missing_images),
            "total_annotations_expected": total_original_annotations,
            "total_annotations_converted": len(self.coco_data['annotations'])
        })
        
        print(f"Images: {len(converted_images)}/{len(original_images)} converted")
        print(f"Annotations: {len(self.coco_data['annotations'])}/{total_original_annotations} converted")
        
        if missing_images:
            print(f"WARNING: {len(missing_images)} images missing from conversion!")
            print(f"Missing images: {list(missing_images)[:5]}...")  # Show first 5
        
        return self.validation_results
    
    def validate_coordinate_conversion(self, sample_count: int = 100) -> Dict:
        """
        Validate coordinate conversion accuracy by sampling
        """
        print(f"Validating coordinate conversion on {sample_count} samples...")
        
        # Create image lookup for faster access
        image_lookup = {img['file_name']: img for img in self.coco_data['images']}
        
        # Group annotations by image
        annotations_by_image = defaultdict(list)
        for ann in self.coco_data['annotations']:
            annotations_by_image[ann['image_id']].append(ann)
        
        # Sample random images for validation
        sample_images = np.random.choice(self.coco_data['images'], 
                                       min(sample_count, len(self.coco_data['images'])), 
                                       replace=False)
        
        coordinate_errors = []
        bbox_out_of_bounds = []
        samples_tested = 0
        
        for img_info in sample_images:
            image_name = img_info['file_name']
            annotation_file = image_name.replace('.png', '.txt').replace('.jpg', '.txt')
            annotation_path = self.yolo_annotations_dir / annotation_file
            
            if not annotation_path.exists():
                continue
            
            # Read original YOLO annotations
            with open(annotation_path, 'r') as f:
                yolo_lines = [line.strip().split() for line in f.readlines() if line.strip()]
            
            # Get corresponding COCO annotations
            coco_annotations = annotations_by_image[img_info['id']]
            
            img_width, img_height = img_info['width'], img_info['height']
            
            # Validate each annotation
            for yolo_line in yolo_lines:
                if len(yolo_line) != 5:
                    continue
                
                yolo_class = int(yolo_line[0])
                yolo_bbox = [float(x) for x in yolo_line[1:5]]
                
                # Convert using reference implementation
                expected_coco_bbox = self.yolo_to_coco_bbox_reference(yolo_bbox, img_width, img_height)
                expected_coco_class = self.yolo_to_coco_class.get(yolo_class)
                
                if expected_coco_class is None:
                    continue
                
                # Find matching COCO annotation
                matching_annotation = None
                for coco_ann in coco_annotations:
                    if coco_ann['category_id'] == expected_coco_class:
                        # Check if bboxes are close (within 1 pixel tolerance)
                        bbox_diff = np.abs(np.array(coco_ann['bbox']) - np.array(expected_coco_bbox))
                        if np.all(bbox_diff < 1.0):  # 1 pixel tolerance
                            matching_annotation = coco_ann
                            break
                
                if matching_annotation:
                    # Validate bbox bounds
                    x, y, w, h = matching_annotation['bbox']
                    if (x < 0 or y < 0 or x + w > img_width or y + h > img_height):
                        bbox_out_of_bounds.append({
                            "image": image_name,
                            "bbox": matching_annotation['bbox'],
                            "image_size": [img_width, img_height]
                        })
                else:
                    # No matching annotation found
                    coordinate_errors.append({
                        "image": image_name,
                        "yolo_bbox": yolo_bbox,
                        "yolo_class": yolo_class,
                        "expected_coco_bbox": expected_coco_bbox,
                        "expected_coco_class": expected_coco_class
                    })
                
                samples_tested += 1

        # Update results
        self.validation_results['coordinate_validation'].update({
            "samples_tested": samples_tested,
            "coordinate_errors": coordinate_errors,
            "bbox_out_of_bounds": bbox_out_of_bounds
        })
        
        print(f"Coordinate validation completed on {samples_tested} samples")
        print(f"Coordinate errors: {len(coordinate_errors)}")
        print(f"Out-of-bounds bboxes: {len(bbox_out_of_bounds)}")
        
        return self.validation_results
    
    def validate_class_mapping(self) -> Dict:
        """
        Validate class mapping correctness
        """
        print("Validating class mapping...")

        # Check if all expected classes are present
        coco_categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        expected_categories = {1: 'door', 2: 'window', 3: 'room'}
        
        mapping_errors = []
        
        # Validate category definitions
        for expected_id, expected_name in expected_categories.items():
            if expected_id not in coco_categories:
                mapping_errors.append(f"Missing category ID {expected_id} ({expected_name})")
            elif coco_categories[expected_id] != expected_name:
                mapping_errors.append(f"Category ID {expected_id}: expected '{expected_name}', got '{coco_categories[expected_id]}'")
        
        # Validate annotation class IDs
        annotation_class_ids = {ann['category_id'] for ann in self.coco_data['annotations']}
        for class_id in annotation_class_ids:
            if class_id not in expected_categories:
                mapping_errors.append(f"Unexpected category ID {class_id} found in annotations")
        
        # Update results
        self.validation_results['class_mapping_validation'].update({
            "correct_mappings": len(expected_categories) - len(mapping_errors),
            "incorrect_mappings": mapping_errors
        })
        
        print(f"Class mapping validation completed")
        print(f"Correct mappings: {len(expected_categories) - len(mapping_errors)}/{len(expected_categories)}")
        if mapping_errors:
            print(f"Mapping errors: {mapping_errors}")
        
        return self.validation_results
    
    def visualize_sample_conversions(self, num_samples: int = 5) -> None:
        """
        Create visualizations comparing original YOLO and converted COCO annotations
        """
        print(f"Creating visualization for {num_samples} samples...")
        
        # Select random samples
        sample_images = np.random.choice(self.coco_data['images'][:100], num_samples, replace=False)

        # Group annotations by image
        annotations_by_image = defaultdict(list)
        for ann in self.coco_data['annotations']:
            annotations_by_image[ann['image_id']].append(ann)
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(15, 5*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for idx, img_info in enumerate(sample_images):
            image_name = img_info['file_name']
            image_path = self.images_dir / image_name
            
            if not image_path.exists():
                continue

            # Load image
            img = cv2.imread(str(image_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_width, img_height = img_info['width'], img_info['height']

            # Plot original image with YOLO annotations 
            axes[idx, 0].imshow(img_rgb)
            axes[idx, 0].set_title(f"Original YOLO: {image_name}")
            axes[idx, 0].axis('off')

            # Read and draw YOLO annotations
            annotation_file = image_name.replace('.png', '.txt').replace('.jpg', '.txt')
            annotation_path = self.yolo_annotations_dir / annotation_file
            
            if annotation_path.exists():
                with open(annotation_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            bbox = [float(x) for x in parts[1:5]]

                            # Convert YOLO to pixel coordinates for visualization
                            center_x = bbox[0] * img_width
                            center_y = bbox[1] * img_height
                            width = bbox[2] * img_width
                            height = bbox[3] * img_height
                            
                            x1 = center_x - width/2
                            y1 = center_y - height/2
                            
                            colors = {0: 'red', 1: 'blue', 2: 'green'}
                            rect = patches.Rectangle((x1, y1), width, height, 
                                                   linewidth=2, edgecolor=colors.get(class_id, 'yellow'), 
                                                   facecolor='none')
                            axes[idx, 0].add_patch(rect)

            # Plot image with COCO annotations
            axes[idx, 1].imshow(img_rgb)
            axes[idx, 1].set_title(f"Converted COCO: {image_name}")
            axes[idx, 1].axis('off')

            # Draw COCO annotations
            coco_annotations = annotations_by_image[img_info['id']]
            colors = {1: 'red', 2: 'blue', 3: 'green'}
            
            for ann in coco_annotations:
                x, y, w, h = ann['bbox']
                class_id = ann['category_id']
                
                rect = patches.Rectangle((x, y), w, h, 
                                       linewidth=2, edgecolor=colors.get(class_id, 'yellow'), 
                                       facecolor='none')
                axes[idx, 1].add_patch(rect)
        
        plt.tight_layout()
        visualization_path = self.output_dir / "conversion_validation_samples.png"
        plt.savefig(visualization_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {visualization_path}")
    
    def generate_validation_report(self) -> str:
        """
        Generate comprehensive validation report
        """
        report = []
        report.append("=" * 60)
        report.append("YOLO TO COCO CONVERSION VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Data completeness
        report.append("1. DATA COMPLETENESS")
        report.append("-" * 30)
        report.append(f"Expected images: {self.validation_results['total_images_expected']}")
        report.append(f"Converted images: {self.validation_results['total_images_converted']}")
        report.append(f"Missing images: {len(self.validation_results['missing_images'])}")
        report.append(f"Expected annotations: {self.validation_results['total_annotations_expected']}")
        report.append(f"Converted annotations: {self.validation_results['total_annotations_converted']}")
        
        if self.validation_results['missing_images']:
            report.append(f"Missing image files: {self.validation_results['missing_images'][:10]}")

        report.append("")
        
        # Coordinate validation
        coord_val = self.validation_results['coordinate_validation']
        report.append("2. COORDINATE CONVERSION")
        report.append("-" * 30)
        report.append(f"Samples tested: {coord_val['samples_tested']}")
        report.append(f"Coordinate errors: {len(coord_val['coordinate_errors'])}")
        report.append(f"Out-of-bounds bboxes: {len(coord_val['bbox_out_of_bounds'])}")

        if coord_val['coordinate_errors']:
            report.append("Sample coordinate errors:")
            for error in coord_val['coordinate_errors'][:3]:  # Show first 3
                report.append(f"  - {error['image']}: YOLO class {error['yolo_class']}")
        
        report.append("")
        
        # Class mapping validation
        class_val = self.validation_results['class_mapping_validation']
        report.append("3. CLASS MAPPING")
        report.append("-" * 30)
        report.append(f"Correct mappings: {class_val['correct_mappings']}")
        report.append(f"Incorrect mappings: {len(class_val['incorrect_mappings'])}")
        
        if class_val['incorrect_mappings']:
            report.append("Mapping errors:")
            for error in class_val['incorrect_mappings']:
                report.append(f"  - {error}")
        
        report.append("")
        
        # Overall assessment
        report.append("4. OVERALL ASSESSMENT")
        report.append("-" * 30)
        
        total_errors = (
            len(self.validation_results['missing_images']) +
            len(coord_val['coordinate_errors']) +
            len(coord_val['bbox_out_of_bounds']) +
            len(class_val['incorrect_mappings'])
        )
        
        if total_errors == 0:
            report.append("CONVERSION SUCCESSFUL - No errors detected")
        else:
            report.append(f"ISSUES DETECTED - Total errors: {total_errors}")
            
            if len(self.validation_results['missing_images']) > 0:
                report.append("  - Missing images need investigation")
            if len(coord_val['coordinate_errors']) > 0:
                report.append("  - Coordinate conversion errors detected")
            if len(coord_val['bbox_out_of_bounds']) > 0:
                report.append("  - Bounding boxes outside image bounds")
            if len(class_val['incorrect_mappings']) > 0:
                report.append("  - Class mapping errors detected")
        
        report.append("")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.output_dir / "validation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"Validation report saved to: {report_path}")
        return report_text
    
    def run_full_validation(self, coordinate_samples: int = 100, visualization_samples: int = 5) -> Dict:
        """
        Run complete validation pipeline
        """
        print("Starting comprehensive validation...")
        
        # Run all validation steps
        self.validate_data_completeness()
        self.validate_coordinate_conversion(coordinate_samples)
        self.validate_class_mapping()
        
        # Create visualizations
        self.visualize_sample_conversions(visualization_samples)
        
        # Generate report
        report = self.generate_validation_report()
        
        # Save detailed results 
        results_path = self.output_dir / "detailed_validation_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print("Validation completed!")
        return self.validation_results


def main():
    """Main function to run validation"""
    parser = argparse.ArgumentParser(description='Validate YOLO to COCO conversion')
    parser.add_argument('--images_dir', type=str, default='data/raw/images',
                       help='Path to original images directory')
    parser.add_argument('--yolo_annotations_dir', type=str, default='data/raw/annotation_file',
                       help='Path to original YOLO annotations directory')  
    parser.add_argument('--coco_file', type=str, default='data/interim/coco_annotations.json',
                       help='Path to converted COCO annotations file')
    parser.add_argument('--output_dir', type=str, default='data/interim/validation_results',
                       help='Directory to save validation results')
    parser.add_argument('--coordinate_samples', type=int, default=100,
                       help='Number of samples for coordinate validation')
    parser.add_argument('--visualization_samples', type=int, default=5,
                       help='Number of samples for visualization')
    
    args = parser.parse_args()
    
    # Run validation
    validator = ConversionValidator(
        original_images_dir=args.images_dir,
        original_annotations_dir=args.yolo_annotations_dir,
        coco_annotations_file=args.coco_file,
        output_dir=args.output_dir
    )
    
    results = validator.run_full_validation(
        coordinate_samples=args.coordinate_samples,
        visualization_samples=args.visualization_samples
    )
    
    return results


if __name__ == "__main__":
    main()