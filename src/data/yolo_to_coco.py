"""
YOLO Format to COCO Format Converter
Converts raw YOLO format annotations (txt files) to COCO format for Faster R-CNN training

Input: YOLO format txt files with format: class_id center_x center_y width height (normalized)
Output: COCO format JSON file
"""

import json
import os
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Tuple
import argparse


class YOLOtoCOCOConverter:
    """
    Converter class for YOLO format to COCO format transformation
    
    YOLO format: class_id center_x center_y width height (normalized 0-1)
    COCO format: [top_left_x, top_left_y, width, height] (absolute pixels)
    """
    
    def __init__(self, images_dir: str, annotations_dir: str, output_dir: str):
        """
        Initialize the converter
        
        Args:
            images_dir: Path to raw images directory
            annotations_dir: Path to YOLO format annotation txt files
            output_dir: Path to output COCO format annotations
        """
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)

        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # COCO format structure
        self.coco_format = {
            "info": {
                "description": "Floor Plan Object Detection Dataset - Converted from YOLO",
                "version": "1.0",
                "year": 2025,
                "contributor": "Buro Happold KTP Associate Interview",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "categories": [
                {"id": 1, "name": "door", "supercategory": "architectural_element"},     # YOLO class 0 -> COCO class 1
                {"id": 2, "name": "window", "supercategory": "architectural_element"},   # YOLO class 1 -> COCO class 2  
                {"id": 3, "name": "room", "supercategory": "space"}                      # YOLO class 2 -> COCO class 3
            ],
            "images": [],
            "annotations": []
        }
        
        # Class mapping from YOLO to COCO (COCO categories start from 1, not 0)
        self.class_mapping = {0: 1, 1: 2, 2: 3}  # door, window, room
        
        self.image_id = 1
        self.annotation_id = 1
    
    def get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """
        Get image width and height from image file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (width, height)
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        height, width = img.shape[:2]
        return width, height
    
    def yolo_to_coco_bbox(self, yolo_bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """
        Convert YOLO bbox format to COCO bbox format / 将YOLO边界框格式转换为COCO边界框格式
        
        YOLO format: [center_x, center_y, width, height] (normalized 0-1)
        YOLO格式：[center_x, center_y, width, height]（归一化0-1）
        
        COCO format: [top_left_x, top_left_y, width, height] (absolute pixels)
        COCO格式：[top_left_x, top_left_y, width, height]（绝对像素）
        
        Args:
            yolo_bbox: YOLO format bounding box / YOLO格式边界框
            img_width: Image width in pixels / 图像宽度（像素）
            img_height: Image height in pixels / 图像高度（像素）
            
        Returns:
            COCO format bounding box / COCO格式边界框
        """
        center_x_norm, center_y_norm, width_norm, height_norm = yolo_bbox
        
        # Convert normalized coordinates to absolute coordinates / 将归一化坐标转换为绝对坐标
        abs_center_x = center_x_norm * img_width
        abs_center_y = center_y_norm * img_height
        abs_width = width_norm * img_width
        abs_height = height_norm * img_height
        
        # Convert center coordinates to top-left coordinates (COCO format)
        # 将中心坐标转换为左上角坐标（COCO格式）
        top_left_x = abs_center_x - abs_width / 2
        top_left_y = abs_center_y - abs_height / 2
        
        return [top_left_x, top_left_y, abs_width, abs_height]
    
    def calculate_area(self, bbox: List[float]) -> float:
        """
        Calculate bounding box area / 计算边界框面积
        
        Args:
            bbox: COCO format bbox [x, y, width, height] / COCO格式边界框
            
        Returns:
            Area in pixels / 面积（像素）
        """
        return bbox[2] * bbox[3]
    
    def process_single_image(self, image_filename: str) -> bool:
        """
        Process a single image and its YOLO format annotations / 处理单张图像及其YOLO格式标注
        
        Args:
            image_filename: Name of the image file / 图像文件名
            
        Returns:
            True if successful, False otherwise / 成功返回True，否则返回False
        """
        # Get image path and corresponding YOLO annotation path / 获取图像路径和对应的YOLO标注路径
        image_path = self.images_dir / image_filename
        annotation_filename = image_filename.replace('.png', '.txt').replace('.jpg', '.txt')
        annotation_path = self.annotations_dir / annotation_filename
        
        # Check if both files exist / 检查两个文件是否都存在
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            return False
        
        if not annotation_path.exists():
            print(f"Warning: YOLO annotation not found: {annotation_path}")
            return False
        
        try:
            # Get image dimensions / 获取图像尺寸
            img_width, img_height = self.get_image_dimensions(str(image_path))
            
            # Add image info to COCO format / 将图像信息添加到COCO格式
            image_info = {
                "id": self.image_id,
                "file_name": image_filename,
                "width": img_width,
                "height": img_height
            }
            self.coco_format["images"].append(image_info)
            
            # Read and process YOLO format annotations / 读取并处理YOLO格式标注
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines / 跳过空行
                    continue
                
                parts = line.split()
                if len(parts) != 5:  # YOLO format: class_id center_x center_y width height
                    print(f"Warning: Invalid YOLO annotation format in {annotation_path}: {line}")
                    continue
                
                try:
                    yolo_class_id = int(parts[0])
                    yolo_bbox = [float(x) for x in parts[1:5]]
                    
                    # Skip if class not in our mapping (0, 1, 2 expected)
                    # 如果类别不在我们的映射中则跳过（期望0, 1, 2）
                    if yolo_class_id not in self.class_mapping:
                        print(f"Warning: Unknown YOLO class {yolo_class_id} in {annotation_path}")
                        continue
                    
                    # Convert YOLO format to COCO format / 将YOLO格式转换为COCO格式
                    coco_class_id = self.class_mapping[yolo_class_id]
                    coco_bbox = self.yolo_to_coco_bbox(yolo_bbox, img_width, img_height)
                    
                    # Calculate area / 计算面积
                    area = self.calculate_area(coco_bbox)
                    
                    # Create COCO annotation entry / 创建COCO标注条目
                    annotation = {
                        "id": self.annotation_id,
                        "image_id": self.image_id,
                        "category_id": coco_class_id,
                        "bbox": coco_bbox,
                        "area": area,
                        "iscrowd": 0
                    }
                    
                    self.coco_format["annotations"].append(annotation)
                    self.annotation_id += 1
                    
                except ValueError as e:
                    print(f"Warning: Error parsing YOLO annotation in {annotation_path}: {line}, Error: {e}")
                    continue
            
            self.image_id += 1
            return True
            
        except Exception as e:
            print(f"Error processing {image_filename}: {e}")
            return False
    
    def convert_dataset(self) -> Dict:
        """
        Convert entire dataset from YOLO format to COCO format
        
        Returns:
            Dictionary with conversion statistics
        """
        print("Starting YOLO to COCO format conversion...")
        
        # Get all image files
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(self.images_dir.glob(f'*{ext}')))
        
        print(f"Found {len(image_files)} images")
        
        # Process each image and its YOLO annotations
        successful_conversions = 0
        failed_conversions = 0
        
        for image_file in image_files:
            if self.process_single_image(image_file.name):
                successful_conversions += 1
            else:
                failed_conversions += 1
        
        # Save COCO format annotations
        output_file = self.output_dir / "1.0-coco_annotations_remove_duplicates.json"
        with open(output_file, 'w') as f:
            json.dump(self.coco_format, f, indent=2)
        
        # Generate conversion statistics
        stats = {
            "total_images": len(image_files),
            "successful_conversions": successful_conversions,
            "failed_conversions": failed_conversions,
            "total_annotations": len(self.coco_format["annotations"]),
            "categories": {
                "door": len([ann for ann in self.coco_format["annotations"] if ann["category_id"] == 1]),
                "window": len([ann for ann in self.coco_format["annotations"] if ann["category_id"] == 2]),
                "room": len([ann for ann in self.coco_format["annotations"] if ann["category_id"] == 3])
            }
        }
        
        print(f"\nYOLO to COCO conversion completed!")
        print(f"Successfully converted: {successful_conversions}/{len(image_files)} images")
        print(f"Total annotations: {stats['total_annotations']}")
        print(f"Class distribution:")
        print(f"  - Doors: {stats['categories']['door']}")
        print(f"  - Windows: {stats['categories']['window']}")
        print(f"  - Rooms: {stats['categories']['room']}")
        print(f"COCO format output saved to: {output_file}")
        
        # Save statistics
        stats_file = self.output_dir / "conversion_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats


def main():
    """
    Main function to run the YOLO to COCO conversion
    """
    parser = argparse.ArgumentParser(description='Convert YOLO format annotations to COCO format')
    parser.add_argument('--images_dir', type=str, 
                       default='data/raw/images',
                       help='Path to raw images directory')
    parser.add_argument('--annotations_dir', type=str,
                       default='data/interim/annotation_file', 
                       help='Path to YOLO format annotation txt files directory')
    parser.add_argument('--output_dir', type=str,
                       default='data/interim',
                       help='Path to output directory for COCO format')
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = YOLOtoCOCOConverter(
        images_dir=args.images_dir,
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir
    )
    
    # Run conversion
    stats = converter.convert_dataset()
    
    return stats


if __name__ == "__main__":
    main()