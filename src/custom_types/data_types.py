"""
Data type definitions for floor plan object detection
为户型图目标检测定义明确的数据类型

This module defines TypedDict classes for clear data format specification
throughout the training, evaluation, and inference pipeline.
"""

from typing import Dict, List, Any, Union, Optional, Tuple, Callable
from typing_extensions import TypedDict, Literal
import torch
import numpy as np

# =============================================================================
# Core Data Format Types
# =============================================================================

class COCOBbox(TypedDict):
    """COCO format bounding box: [x, y, width, height]"""
    x: float
    y: float 
    width: float
    height: float

class PascalVOCBbox(TypedDict):
    """Pascal VOC format bounding box: [x1, y1, x2, y2]"""
    x1: float
    y1: float
    x2: float
    y2: float

# Coordinate formats
COCOBboxArray = Tuple[float, float, float, float]  # (x, y, width, height)
PascalVOCBboxArray = Tuple[float, float, float, float]  # (x1, y1, x2, y2)

# =============================================================================
# COCO Annotation Types
# =============================================================================

class COCOAnnotation(TypedDict):
    """Single COCO format annotation"""
    id: int
    image_id: int
    category_id: int
    bbox: COCOBboxArray  # [x, y, width, height]
    area: float
    iscrowd: int  # 0 or 1

class COCOImage(TypedDict):
    """COCO format image metadata"""
    id: int
    width: int
    height: int
    file_name: str

class COCOCategory(TypedDict):
    """COCO format category definition"""
    id: int
    name: str
    supercategory: str

class COCODataset(TypedDict):
    """Complete COCO format dataset structure"""
    images: List[COCOImage]
    annotations: List[COCOAnnotation]
    categories: List[COCOCategory]

# =============================================================================
# Model Input/Output Types
# =============================================================================

class ModelTarget(TypedDict):
    """PyTorch model target format (Pascal VOC coordinates)"""
    boxes: torch.Tensor  # Shape: [N, 4], format: [x1, y1, x2, y2]
    labels: torch.Tensor  # Shape: [N], category IDs
    area: torch.Tensor  # Shape: [N], bounding box areas
    iscrowd: torch.Tensor  # Shape: [N], crowd flags
    image_id: torch.Tensor  # Shape: [1], image ID

class ModelPrediction(TypedDict):
    """PyTorch model prediction format (Pascal VOC coordinates)"""
    boxes: torch.Tensor  # Shape: [N, 4], format: [x1, y1, x2, y2]
    scores: torch.Tensor  # Shape: [N], confidence scores
    labels: torch.Tensor  # Shape: [N], predicted category IDs

# =============================================================================
# Dataset Sample Types
# =============================================================================

class DatasetSample(TypedDict):
    """Single dataset sample with all metadata"""
    image: torch.Tensor  # Shape: [C, H, W], preprocessed image tensor
    target: ModelTarget  # Ground truth annotations in model format
    image_id: int  # Original image ID
    image_info: COCOImage  # Original image metadata
    original_size: Tuple[int, int]  # (height, width) before preprocessing

class DatasetBatch(TypedDict):
    """Batch of dataset samples"""
    images: List[torch.Tensor]  # List of [C, H, W] tensors
    targets: List[ModelTarget]  # List of ground truth targets

# =============================================================================
# Evaluation Types
# =============================================================================

class COCOPrediction(TypedDict):
    """COCO format prediction for evaluation"""
    image_id: int
    category_id: int
    bbox: COCOBboxArray  # [x, y, width, height] in COCO format
    score: float

class EvaluationMetrics(TypedDict):
    """Comprehensive evaluation metrics"""
    mAP: float  # mAP@[0.5:0.95]
    mAP_50: float  # mAP@0.5
    mAP_75: float  # mAP@0.75
    mAP_small: float  # mAP for small objects
    mAP_medium: float  # mAP for medium objects
    mAP_large: float  # mAP for large objects
    mAR_1: float  # mAR with max 1 detection
    mAR_10: float  # mAR with max 10 detections
    mAR_100: float  # mAR with max 100 detections
    mAR_small: float  # mAR for small objects
    mAR_medium: float  # mAR for medium objects
    mAR_large: float  # mAR for large objects

class PerCategoryMetrics(TypedDict):
    """Per-category evaluation metrics"""
    AP_doors: float
    AP_windows: float  
    AP_rooms: float
    AP50_doors: float
    AP50_windows: float
    AP50_rooms: float

# Combined metrics type
class CompleteMetrics(EvaluationMetrics, PerCategoryMetrics):
    """Complete evaluation metrics including per-category results"""
    pass

# =============================================================================
# Training Types
# =============================================================================

class TrainingStep(TypedDict):
    """Single training step data"""
    epoch: int
    batch_idx: int
    total_batches: int
    loss: float
    loss_components: Dict[str, float]  # e.g., {'loss_classifier': 0.5, 'loss_box_reg': 0.3}

class EpochResults(TypedDict):
    """Results from one training epoch"""
    epoch: int
    train_metrics: Dict[str, float]  # Training losses
    val_metrics: Dict[str, float]  # Validation metrics
    learning_rate: float
    epoch_time: float

class TrainingHistory(TypedDict):
    """Complete training history"""
    epochs: List[EpochResults]
    best_epoch: int
    best_mAP: float
    total_training_time: float

# =============================================================================
# Configuration Types
# =============================================================================

class ModelConfig(TypedDict):
    """Model configuration"""
    architecture: Literal['faster_rcnn', 'retinanet']
    backbone: Literal['resnet50', 'resnet101', 'mobilenet_v3_large', 'mobilenet_v3_small']
    num_classes: int
    pretrained: bool
    trainable_backbone_layers: int

class DataConfig(TypedDict):
    """Data configuration"""
    images_dir: str
    annotations_file: str
    splits_dir: str
    batch_size: int
    num_workers: int

class TrainingConfig(TypedDict):
    """Training configuration"""
    num_epochs: int
    learning_rate: float
    weight_decay: float
    gradient_clipping: float
    save_every: int
    validate_every: int

# =============================================================================
# Transformation Types
# =============================================================================

class AugmentationResult(TypedDict):
    """Result from data augmentation"""
    image: np.ndarray  # Augmented image
    bboxes: List[PascalVOCBboxArray]  # Augmented bounding boxes
    category_ids: List[int]  # Category IDs (unchanged)

class TransformResult(TypedDict):
    """Result from preprocessing transforms"""
    image: torch.Tensor  # Preprocessed image tensor
    bboxes: Optional[List[PascalVOCBboxArray]]  # Transformed bounding boxes
    category_ids: Optional[List[int]]  # Category IDs

# =============================================================================
# Utility Types
# =============================================================================

# Device type
Device = Union[torch.device, str]

# Image size type  
ImageSize = Tuple[int, int]  # (height, width)

# Color type for preprocessing
Color = Tuple[int, int, int]  # RGB values

# Coordinate conversion function types
CoordinateConverter = Callable[[Union[COCOBboxArray, PascalVOCBboxArray]], 
                             Union[PascalVOCBboxArray, COCOBboxArray]]

# =============================================================================
# Validation Types
# =============================================================================

class ValidationResult(TypedDict):
    """Data pipeline validation result"""
    coordinate_conversion_correct: bool
    format_consistency_check: bool
    sample_count: int
    error_messages: List[str]

class CoordinateValidation(TypedDict):
    """Coordinate transformation validation"""
    original_coco: COCOBboxArray
    converted_pascal: PascalVOCBboxArray  
    reconverted_coco: COCOBboxArray
    conversion_error: float
    is_valid: bool

# =============================================================================
# Export all types
# =============================================================================

__all__ = [
    # Core formats
    'COCOBbox', 'PascalVOCBbox', 'COCOBboxArray', 'PascalVOCBboxArray',
    
    # COCO types
    'COCOAnnotation', 'COCOImage', 'COCOCategory', 'COCODataset',
    
    # Model types
    'ModelTarget', 'ModelPrediction', 
    
    # Dataset types
    'DatasetSample', 'DatasetBatch',
    
    # Evaluation types
    'COCOPrediction', 'EvaluationMetrics', 'PerCategoryMetrics', 'CompleteMetrics',
    
    # Training types
    'TrainingStep', 'EpochResults', 'TrainingHistory',
    
    # Configuration types
    'ModelConfig', 'DataConfig', 'TrainingConfig',
    
    # Transformation types
    'AugmentationResult', 'TransformResult',
    
    # Utility types
    'Device', 'ImageSize', 'Color', 'CoordinateConverter',
    
    # Validation types
    'ValidationResult', 'CoordinateValidation'
]