"""
Unified data preprocessing pipeline for floor plan object detection
统一的户型图目标检测数据预处理管道

This module provides a single, consistent pipeline that handles:
- Data augmentation (rotation, flipping)  
- Preprocessing (resize, padding, normalization)
- Format conversion (numpy to tensor)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


def create_unified_transform(config: Dict[str, Any], is_training: bool = False) -> A.Compose:
    """
    Create a unified Albumentations pipeline that handles all preprocessing
    创建统一的Albumentations管道处理所有预处理
    
    Args:
        config: Configuration dictionary
        is_training: Whether this is for training (enables augmentation)
        
    Returns:
        Albumentations Compose pipeline
    """
    
    # Get configuration parameters
    preprocessing_config = config.get('preprocessing', {})
    augmentation_config = config.get('augmentation', {})
    
    # Base preprocessing parameters
    target_size = preprocessing_config.get('target_size', 800)
    pad_color = preprocessing_config.get('pad_color', [128, 128, 128])
    
    # Normalization parameters (ImageNet defaults for pretrained models)
    normalize_mean = preprocessing_config.get('normalize_mean', [0.485, 0.456, 0.406])
    normalize_std = preprocessing_config.get('normalize_std', [0.229, 0.224, 0.225])
    
    # CLAHE parameters
    apply_clahe = preprocessing_config.get('apply_clahe', True)
    clahe_clip_limit = preprocessing_config.get('clahe_clip_limit', 2.0)
    clahe_tile_grid_size = tuple(preprocessing_config.get('clahe_tile_grid_size', [8, 8]))
    
    # Build pipeline transforms list
    transforms = []
    
    # 1. Optional CLAHE enhancement (before augmentation)
    if apply_clahe:
        transforms.append(
            A.CLAHE(
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size,
                p=1.0
            )
        )
    
    # 2. Data augmentation (only if training and enabled)
    if is_training:
        # Rotation augmentation
        if augmentation_config.get('apply_rotation', False):
            rotation_prob = augmentation_config.get('rotation_probability', 0.3)
            transforms.append(
                A.RandomRotate90(p=rotation_prob)
            )
        
        # Horizontal flip
        if augmentation_config.get('apply_flip', False):
            h_flip_prob = augmentation_config.get('horizontal_flip_probability', 0.5)
            transforms.append(
                A.HorizontalFlip(p=h_flip_prob)
            )
            
            # Vertical flip (usually less common for floor plans)
            v_flip_prob = augmentation_config.get('vertical_flip_probability', 0.1)
            if v_flip_prob > 0:
                transforms.append(
                    A.VerticalFlip(p=v_flip_prob)
                )
    
    # 3. Resize with aspect ratio preservation and padding
    transforms.append(
        A.LongestMaxSize(max_size=target_size, interpolation=cv2.INTER_LINEAR)
    )
    
    transforms.append(
        A.PadIfNeeded(
            min_height=target_size,
            min_width=target_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=pad_color,
            p=1.0
        )
    )
    
    # 4. Normalization and tensor conversion
    transforms.append(
        A.Normalize(
            mean=normalize_mean,
            std=normalize_std,
            max_pixel_value=255.0,
            p=1.0
        )
    )
    
    transforms.append(ToTensorV2(p=1.0))
    
    # Create the complete pipeline
    pipeline = A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',  # (x1, y1, x2, y2)
            label_fields=['category_ids'],
            min_area=0,
            min_visibility=0,
            check_each_transform=False  # Disable for performance
        )
    )
    
    logger.info(f"Created unified transform pipeline with {len(transforms)} transforms")
    logger.info(f"Training mode: {is_training}, Target size: {target_size}")
    
    return pipeline


def create_simple_inference_transform(target_size: int = 800) -> A.Compose:
    """
    Create a simple transform pipeline for inference only
    创建仅用于推理的简单变换管道
    
    Args:
        target_size: Target image size
        
    Returns:
        Simple Albumentations pipeline for inference
    """
    
    transforms = [
        # Resize with padding
        A.LongestMaxSize(max_size=target_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=target_size,
            min_width=target_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=[128, 128, 128],
            p=1.0
        ),
        
        # Normalize and convert to tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2(p=1.0)
    ]
    
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['category_ids'],
            min_area=0,
            min_visibility=0
        )
    )


def apply_unified_transform(transform_pipeline: A.Compose, 
                          image: np.ndarray, 
                          bboxes: Optional[List[List[float]]] = None,
                          category_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Apply unified transform pipeline to image and annotations
    对图像和标注应用统一变换管道
    
    Args:
        transform_pipeline: Albumentations compose pipeline
        image: Input image as numpy array (H, W, C)
        bboxes: List of bounding boxes in Pascal VOC format [[x1, y1, x2, y2], ...]
        category_ids: List of category IDs corresponding to bboxes
        
    Returns:
        Dictionary with transformed 'image', 'bboxes', and 'category_ids'
    """
    
    try:
        # Prepare inputs
        if bboxes is None:
            bboxes = []
        if category_ids is None:
            category_ids = []
            
        # Ensure bboxes and category_ids have same length
        if len(bboxes) != len(category_ids):
            logger.warning(f"Bboxes ({len(bboxes)}) and category_ids ({len(category_ids)}) length mismatch")
            min_len = min(len(bboxes), len(category_ids))
            bboxes = bboxes[:min_len]
            category_ids = category_ids[:min_len]
        
        # Apply transforms
        transformed = transform_pipeline(
            image=image,
            bboxes=bboxes,
            category_ids=category_ids
        )
        
        return transformed
        
    except Exception as e:
        logger.error(f"Transform application failed: {e}")
        logger.error(f"Image shape: {image.shape}, Bboxes: {len(bboxes) if bboxes else 0}")
        
        # Return fallback: simple resize and normalize
        try:
            fallback_transform = create_simple_inference_transform()
            transformed = fallback_transform(
                image=image,
                bboxes=bboxes or [],
                category_ids=category_ids or []
            )
            logger.warning("Applied fallback transform")
            return transformed
        except Exception as fallback_error:
            logger.error(f"Fallback transform also failed: {fallback_error}")
            raise


def validate_transform_output(transformed_data: Dict[str, Any]) -> bool:
    """
    Validate the output of transform pipeline
    
    Args:
        transformed_data: Output from transform pipeline
        
    Returns:
        True if output is valid
    """
    
    try:
        # Check required keys
        if 'image' not in transformed_data:
            logger.error("Missing 'image' key in transformed data")
            return False
            
        image = transformed_data['image']
        
        # Check image is tensor with correct shape
        if not hasattr(image, 'shape') or len(image.shape) != 3:
            logger.error(f"Invalid image tensor shape: {getattr(image, 'shape', 'unknown')}")
            return False
            
        # Check image has 3 channels (RGB)
        if image.shape[0] != 3:
            logger.error(f"Invalid number of channels: {image.shape[0]}, expected 3")
            return False
            
        # Check bboxes format if present
        if 'bboxes' in transformed_data and transformed_data['bboxes']:
            bboxes = transformed_data['bboxes']
            for i, bbox in enumerate(bboxes):
                if len(bbox) != 4:
                    logger.error(f"Bbox {i} has wrong format: {bbox}, expected [x1, y1, x2, y2]")
                    return False
                x1, y1, x2, y2 = bbox
                if x2 <= x1 or y2 <= y1:
                    logger.error(f"Invalid bbox {i}: {bbox}")
                    return False
                    
        logger.debug("Transform output validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Transform validation failed: {e}")
        return False


# Convenience functions for common use cases
def create_training_transform(config: Dict[str, Any]) -> A.Compose:
    """Create transform pipeline for training with augmentation"""
    return create_unified_transform(config, is_training=True)


def create_validation_transform(config: Dict[str, Any]) -> A.Compose:
    """Create transform pipeline for validation without augmentation"""  
    return create_unified_transform(config, is_training=False)