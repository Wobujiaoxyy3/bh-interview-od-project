"""
COCO Dataset classes for floor plan object detection
Handles loading, preprocessing, and augmentation of COCO format data
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from .unified_transforms import create_unified_transform, apply_unified_transform

# Import type definitions
try:
    from ..custom_types import DatasetSample
except ImportError:
    # Handle direct script execution
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from custom_types import DatasetSample


logger = logging.getLogger(__name__)


class FloorPlanDataset(Dataset[DatasetSample]):
    """
    Dataset class for floor plan object detection with COCO format annotations
    """

    # Type annotations for instance variables   
    annotations_file: Path
    images_dir: Path
    config: Dict[str, Any]
    is_training: bool
    use_augmentation: bool
    coco: COCO
    image_ids: List[int]
    categories: Dict[int, Dict[str, Any]]
    category_names: Dict[int, str]
    num_classes: int
    transform_pipeline: Optional[Callable]
    
    def __init__(self, 
                 annotations_file: Union[str, Path],
                 images_dir: Union[str, Path],
                 config: Dict[str, Any],
                 is_training: bool = False,
                 use_augmentation: bool = False) -> None:
        """
        Initialize dataset
        
        Args:
            annotations_file: Path to COCO format annotations JSON file
            images_dir: Directory containing images
            config: Configuration dictionary with preprocessing/augmentation settings
            is_training: Whether this is training dataset
            use_augmentation: Whether to apply data augmentation
        """
        self.annotations_file = Path(annotations_file)
        self.images_dir = Path(images_dir)
        self.config = config
        self.is_training = is_training
        self.use_augmentation = use_augmentation and is_training
        
        # Load COCO annotations
        self.coco = COCO(str(self.annotations_file))
        self.image_ids = list(self.coco.imgs.keys())
        
        # Get category information
        self.categories = {cat['id']: cat for cat in self.coco.loadCats(self.coco.getCatIds())}
        self.category_names = {cat_id: cat['name'] for cat_id, cat in self.categories.items()}
        self.num_classes = len(self.categories)
        
        # Create unified transform pipeline
        # This handles both augmentation (if training) and preprocessing
        self.transform_pipeline = create_unified_transform(
            config=config,
            is_training=is_training and self.use_augmentation
        )
        
        logger.info(f"Initialized FloorPlanDataset with {len(self.image_ids)} images, "
                   f"{self.num_classes} classes, training={is_training}, "
                   f"augmentation={self.use_augmentation}")
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> DatasetSample:
        """
        Get item by index
        
        Args:
            idx: Index
            
        Returns:
            Dictionary containing image, target, and metadata
        """
        image_id = self.image_ids[idx]
        
        # Load image
        image_info = self.coco.imgs[image_id]
        image_path = self.images_dir / image_info['file_name']
        
        try:
            # Load image using PIL and convert to RGB
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return dummy data
            return self._get_dummy_sample(idx)
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Parse annotations to get boxes and labels
        boxes, labels = self._parse_annotations_simple(annotations)
        
        # Apply unified transform pipeline (handles augmentation + preprocessing)
        transformed = self._apply_unified_transforms(image, boxes, labels)
        
        # Update target with correct image_id
        transformed['target']['image_id'] = torch.tensor([image_id], dtype=torch.int64)
        
        # Calculate transformation parameters for coordinate conversion
        original_height, original_width = image.shape[:2]
        
        # Get preprocessing config
        preprocessing_config = self.config.get('preprocessing', {})
        target_size = preprocessing_config.get('target_size', 800)
        
        # Calculate scale and padding
        scale = min(target_size / original_width, target_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Calculate padding
        pad_left = (target_size - new_width) // 2
        pad_top = (target_size - new_height) // 2
        
        # Store transformation info in target
        transformed['target']['scale'] = torch.tensor(scale, dtype=torch.float32)
        transformed['target']['pad_left'] = torch.tensor(pad_left, dtype=torch.float32)
        transformed['target']['pad_top'] = torch.tensor(pad_top, dtype=torch.float32)
        transformed['target']['original_size'] = torch.tensor([original_height, original_width], dtype=torch.int64)
        
        return {
            'image': transformed['image'],
            'target': transformed['target'],
            'image_id': image_id,
            'image_info': image_info,
            'original_size': (image_info['height'], image_info['width'])
        }
    
    def _parse_annotations_simple(self, annotations: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[int]]:
        """
        Parse COCO annotations into simple format for unified transforms
        
        Args:
            annotations: List of COCO annotations
            
        Returns:
            Tuple of (boxes_list, labels_list) where boxes are in Pascal VOC format
        """
        boxes: List[List[float]] = []
        labels: List[int] = []
        
        for ann in annotations:
            # Convert COCO bbox format (x, y, width, height) to Pascal VOC (x1, y1, x2, y2)
            x, y, w, h = ann['bbox']
            pascal_bbox = [float(x), float(y), float(x + w), float(y + h)]
            boxes.append(pascal_bbox)
            labels.append(ann['category_id'])
        
        return boxes, labels
    
    def _apply_unified_transforms(self, image: np.ndarray, boxes: List[List[float]], labels: List[int]) -> Dict[str, Any]:
        """
        Apply unified transform pipeline (augmentation + preprocessing)
        
        Args:
            image: Input image as numpy array
            boxes: List of bounding boxes in Pascal VOC format
            labels: List of category labels
            
        Returns:
            Dictionary with transformed image and target tensors
        """
        try:
            # Apply unified transform pipeline
            transformed = apply_unified_transform(
                transform_pipeline=self.transform_pipeline,
                image=image,
                bboxes=boxes,
                category_ids=labels
            )
            
            # Convert to final model format
            transformed_boxes = transformed.get('bboxes', [])
            transformed_labels = transformed.get('category_ids', [])
            
            if transformed_boxes and len(transformed_boxes) > 0:
                # Convert to tensors
                boxes_tensor = torch.as_tensor(transformed_boxes, dtype=torch.float32)
                labels_tensor = torch.as_tensor(transformed_labels, dtype=torch.int64)
                
                # Calculate areas
                areas = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
                
                target = {
                    'boxes': boxes_tensor,
                    'labels': labels_tensor,
                    'area': areas,
                    'iscrowd': torch.zeros(len(transformed_boxes), dtype=torch.int64),
                    'image_id': torch.tensor([0], dtype=torch.int64)  # Will be updated by caller
                }
            else:
                # Empty target for images without annotations
                target = {
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),
                    'labels': torch.zeros(0, dtype=torch.int64),
                    'area': torch.zeros(0, dtype=torch.float32),
                    'iscrowd': torch.zeros(0, dtype=torch.int64),
                    'image_id': torch.tensor([0], dtype=torch.int64)  # Will be updated by caller
                }
            
            return {
                'image': transformed['image'],
                'target': target
            }
            
        except Exception as e:
            logger.error(f"Unified transform failed: {e}. Using fallback.")
            # Fallback: simple tensor conversion
            try:
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                
                if boxes and len(boxes) > 0:
                    boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
                    labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
                    areas = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
                    
                    target = {
                        'boxes': boxes_tensor,
                        'labels': labels_tensor,
                        'area': areas,
                        'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
                        'image_id': torch.tensor([0], dtype=torch.int64)
                    }
                else:
                    target = {
                        'boxes': torch.zeros((0, 4), dtype=torch.float32),
                        'labels': torch.zeros(0, dtype=torch.int64),
                        'area': torch.zeros(0, dtype=torch.float32),
                        'iscrowd': torch.zeros(0, dtype=torch.int64),
                        'image_id': torch.tensor([0], dtype=torch.int64)
                    }
                
                return {
                    'image': image_tensor,
                    'target': target
                }
            except Exception as fallback_error:
                logger.error(f"Fallback transform also failed: {fallback_error}")
                raise
    
    def _get_dummy_sample(self, idx: int) -> Dict[str, Any]:
        """
        Get dummy sample for failed image loading
        
        Args:
            idx: Index
            
        Returns:
            Dummy sample dictionary
        """
        dummy_image = torch.zeros(3, 800, 800)
        dummy_target = {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros(0, dtype=torch.int64),
            'area': torch.zeros(0, dtype=torch.float32),
            'iscrowd': torch.zeros(0, dtype=torch.int64),
            'image_id': torch.tensor([self.image_ids[idx]], dtype=torch.int64)
        }
        
        return {
            'image': dummy_image,
            'target': dummy_target,
            'image_id': self.image_ids[idx],
            'image_info': {'id': self.image_ids[idx], 'width': 800, 'height': 800},
            'original_size': (800, 800)
        }
    
    def get_category_mapping(self) -> Dict[int, str]:
        """Get category ID to name mapping"""
        return self.category_names.copy()
    
    def get_image_path(self, image_id: int) -> Path:
        """Get image path for given image ID"""
        image_info = self.coco.imgs[image_id]
        return self.images_dir / image_info['file_name']


def collate_fn(batch: List[Dict[str, Any]]) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """
    Custom collate function for object detection batches
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Tuple of (images, targets)
    """
    images = []
    targets = []
    
    for sample in batch:
        images.append(sample['image'])
        targets.append(sample['target'])
    
    return images, targets


def create_datasets(config: Dict[str, Any]) -> Tuple[FloorPlanDataset, FloorPlanDataset, FloorPlanDataset]:
    """
    Create train, validation, and test datasets
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_config = config.get('data', {})
    
    # Dataset paths
    images_dir = Path(data_config.get('images_dir', 'data/raw/images'))
    splits_dir = Path(data_config.get('splits_dir', 'data/processed/splits'))
    
    # Create datasets with augmentation enabled for training
    train_dataset = FloorPlanDataset(
        annotations_file=splits_dir / 'train_annotations.json',
        images_dir=images_dir,
        config=config,
        is_training=True,
        use_augmentation=config.get('augmentation', {}).get('apply_rotation', False) or 
                        config.get('augmentation', {}).get('apply_flip', False)  # Enable if configured
    )
    
    val_dataset = FloorPlanDataset(
        annotations_file=splits_dir / 'val_annotations.json',
        images_dir=images_dir,
        config=config,
        is_training=False,
        use_augmentation=False
    )
    
    test_dataset = FloorPlanDataset(
        annotations_file=splits_dir / 'test_annotations.json',
        images_dir=images_dir,
        config=config,
        is_training=False,
        use_augmentation=False
    )
    
    logger.info(f"Created datasets - Train: {len(train_dataset)}, "
               f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(config: Dict[str, Any]) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create data loaders for train, validation, and test datasets
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of data loaders
    """
    train_dataset, val_dataset, test_dataset = create_datasets(config)
    
    # Data loader configuration
    loader_config = config.get('data_loader', {})
    batch_size = loader_config.get('batch_size', 4)
    num_workers = loader_config.get('num_workers', 4)
    pin_memory = loader_config.get('pin_memory', True)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False 
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader