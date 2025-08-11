"""
RetinaNet model implementation for floor plan object detection
Single-stage detector with Feature Pyramid Network and Focal Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import torchvision
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
import math
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss
        
        Args:
            alpha: Weighting factor for rare class (typical: 0.25)
            gamma: Focusing parameter (typical: 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            inputs: Predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class RetinaNetHead(nn.Module):
    """
    RetinaNet classification and regression head
    """
    
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int, 
                 num_layers: int = 4, prior_probability: float = 0.01):
        """
        Initialize RetinaNet head
        
        Args:
            in_channels: Number of input channels from FPN
            num_anchors: Number of anchors per location
            num_classes: Number of classes (including background)
            num_layers: Number of convolutional layers in head
            prior_probability: Prior probability for focal loss initialization
        """
        super().__init__()
        
        # Classification head
        cls_layers = []
        for _ in range(num_layers):
            cls_layers.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            cls_layers.append(nn.ReLU())
        cls_layers.append(nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1))
        self.cls_head = nn.Sequential(*cls_layers)
        
        # Regression head
        reg_layers = []
        for _ in range(num_layers):
            reg_layers.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            reg_layers.append(nn.ReLU())
        reg_layers.append(nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1))
        self.reg_head = nn.Sequential(*reg_layers)
        
        # Initialize weights
        self._init_weights(prior_probability)
    
    def _init_weights(self, prior_probability: float):
        """Initialize head weights"""
        for layer in self.cls_head:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)
        
        for layer in self.reg_head:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)
        
        # Initialize classification layer bias for focal loss
        bias_value = -math.log((1 - prior_probability) / prior_probability)
        nn.init.constant_(self.cls_head[-1].bias, bias_value)
    
    def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            features: List of feature maps from FPN
            
        Returns:
            Tuple of (classifications, regressions)
        """
        classifications = []
        regressions = []
        
        for feature in features:
            cls_output = self.cls_head(feature)
            reg_output = self.reg_head(feature)
            
            classifications.append(cls_output)
            regressions.append(reg_output)
        
        return classifications, regressions


class FloorPlanRetinaNet(nn.Module):
    """
    Custom RetinaNet implementation optimized for floor plan object detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RetinaNet model
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        
        # Model parameters
        self.num_classes = config.get('num_classes', 3)  # Background + floor plan classes
        self.backbone_name = config.get('backbone', 'resnet50')
        self.pretrained_backbone = config.get('pretrained_backbone', True)
        self.trainable_backbone_layers = config.get('trainable_backbone_layers', 3)
        
        # Anchor parameters
        self.anchor_sizes = config.get('anchor_sizes', ((32, 40, 50), (64, 80, 101), 
                                                       (128, 161, 203), (256, 322, 406), 
                                                       (512, 645, 813)))
        self.aspect_ratios = config.get('aspect_ratios', ((0.5, 1.0, 2.0),) * len(self.anchor_sizes))
        
        # Head parameters
        self.num_head_layers = config.get('num_head_layers', 4)
        self.prior_probability = config.get('prior_probability', 0.01)
        
        # Loss parameters
        self.focal_loss_alpha = config.get('focal_loss_alpha', 0.25)
        self.focal_loss_gamma = config.get('focal_loss_gamma', 2.0)
        
        # Detection parameters
        self.score_threshold = config.get('score_threshold', 0.05)
        self.nms_threshold = config.get('nms_threshold', 0.5)
        self.max_detections = config.get('max_detections', 100)
        
        # Build model
        self._build_model()
        
        logger.info(f"Initialized FloorPlanRetinaNet with {self.num_classes} classes, "
                   f"backbone: {self.backbone_name}")
    
    def _build_model(self):
        """Build the RetinaNet model components"""
        
        # Create backbone with FPN
        self.backbone = resnet_fpn_backbone(
            backbone_name=self.backbone_name,
            pretrained=self.pretrained_backbone,
            trainable_layers=self.trainable_backbone_layers
        )
        
        # Create anchor generator
        self.anchor_generator = AnchorGenerator(
            sizes=self.anchor_sizes,
            aspect_ratios=self.aspect_ratios
        )
        
        # Create RetinaNet head
        self.head = RetinaNetHead(
            in_channels=self.backbone.out_channels,
            num_anchors=self.anchor_generator.num_anchors_per_location()[0],
            num_classes=self.num_classes,
            num_layers=self.num_head_layers,
            prior_probability=self.prior_probability
        )
        
        # Create focal loss
        self.focal_loss = FocalLoss(
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma
        )
    
    def forward(self, images: List[torch.Tensor], 
                targets: Optional[List[Dict[str, torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: List of images (each is a tensor of shape [C, H, W])
            targets: List of targets (for training)
            
        Returns:
            Dictionary with losses (training) or predictions (inference)
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets must be provided")
        
        # Extract features using backbone
        features = self.backbone(torch.stack(images))
        if isinstance(features, torch.Tensor):
            features = [features]
        elif isinstance(features, dict):
            features = list(features.values())
        
        # Get predictions from head
        classifications, regressions = self.head(features)
        
        if self.training:
            # Compute losses during training
            return self._compute_losses(classifications, regressions, targets, images)
        else:
            # Generate predictions during inference
            return self._generate_predictions(classifications, regressions, images)
    
    def _compute_losses(self, classifications: List[torch.Tensor], 
                       regressions: List[torch.Tensor],
                       targets: List[Dict[str, torch.Tensor]], 
                       images: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute training losses
        
        Args:
            classifications: Classification predictions
            regressions: Regression predictions
            targets: Ground truth targets
            images: Input images
            
        Returns:
            Dictionary of losses
        """
        # This is a simplified implementation
        # In practice, you would need proper anchor matching and loss computation
        
        # Placeholder losses
        classification_loss = torch.tensor(0.0, device=classifications[0].device, requires_grad=True)
        regression_loss = torch.tensor(0.0, device=regressions[0].device, requires_grad=True)
        
        return {
            'classification_loss': classification_loss,
            'bbox_regression_loss': regression_loss,
            'loss': classification_loss + regression_loss
        }
    
    def _generate_predictions(self, classifications: List[torch.Tensor],
                            regressions: List[torch.Tensor],
                            images: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Generate predictions during inference
        
        Args:
            classifications: Classification predictions
            regressions: Regression predictions
            images: Input images
            
        Returns:
            List of prediction dictionaries
        """
        # Placeholder implementation
        # In practice, you would decode anchors, apply NMS, etc.
        
        predictions = []
        for _ in images:
            pred = {
                'boxes': torch.zeros((0, 4)),
                'labels': torch.zeros(0, dtype=torch.int64),
                'scores': torch.zeros(0)
            }
            predictions.append(pred)
        
        return predictions
    
    def predict(self, images: List[torch.Tensor], 
                score_threshold: float = 0.5) -> List[Dict[str, torch.Tensor]]:
        """
        Make predictions on images
        
        Args:
            images: List of input images
            score_threshold: Score threshold for filtering predictions
            
        Returns:
            List of prediction dictionaries
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(images)
        
        # Filter by score threshold
        filtered_predictions = []
        for pred in predictions:
            if len(pred['scores']) > 0:
                mask = pred['scores'] > score_threshold
                filtered_pred = {
                    'boxes': pred['boxes'][mask],
                    'labels': pred['labels'][mask],
                    'scores': pred['scores'][mask]
                }
            else:
                filtered_pred = pred
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions


class SimpleRetinaNet(nn.Module):
    """
    Simplified RetinaNet using torchvision's pre-built model
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize simplified RetinaNet
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        self.num_classes = config.get('num_classes', 3)
        
        # Load pre-trained model
        model_name = config.get('model_name', 'retinanet_resnet50_fpn')
        pretrained = config.get('pretrained', True)
        
        if model_name == 'retinanet_resnet50_fpn':
            self.model = torchvision.models.detection.retinanet_resnet50_fpn(
                pretrained=pretrained,
                num_classes=self.num_classes,
                score_thresh=config.get('score_threshold', 0.05),
                nms_thresh=config.get('nms_threshold', 0.5),
                detections_per_img=config.get('max_detections', 100)
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Update focal loss parameters if specified
        if hasattr(self.model.head, 'classification_head'):
            if hasattr(self.model.head.classification_head, 'focal_loss'):
                focal_loss_alpha = config.get('focal_loss_alpha', 0.25)
                focal_loss_gamma = config.get('focal_loss_gamma', 2.0)
                self.model.head.classification_head.focal_loss.alpha = focal_loss_alpha
                self.model.head.classification_head.focal_loss.gamma = focal_loss_gamma
        
        logger.info(f"Initialized SimpleRetinaNet with {self.num_classes} classes, "
                   f"model: {model_name}")
    
    def forward(self, images: List[torch.Tensor], 
                targets: Optional[List[Dict[str, torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        return self.model(images, targets)
    
    def predict(self, images: List[torch.Tensor], 
                score_threshold: float = 0.5) -> List[Dict[str, torch.Tensor]]:
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)
        
        # Filter by score threshold
        filtered_predictions = []
        for pred in predictions:
            if len(pred['scores']) > 0:
                mask = pred['scores'] > score_threshold
                filtered_pred = {
                    'boxes': pred['boxes'][mask],
                    'labels': pred['labels'][mask],
                    'scores': pred['scores'][mask]
                }
            else:
                filtered_pred = pred
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions


def create_retinanet_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create RetinaNet model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RetinaNet model
    """
    model_type = config.get('model_type', 'simple')
    
    if model_type == 'simple':
        return SimpleRetinaNet(config)
    elif model_type == 'custom':
        return FloorPlanRetinaNet(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_retinanet_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """
    Load model from checkpoint
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info(f"Loaded RetinaNet checkpoint from {checkpoint_path}")
    return model