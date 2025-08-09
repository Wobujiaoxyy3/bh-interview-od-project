"""
Faster R-CNN model implementation for floor plan object detection
Fixed version addressing thread-safety, modern API usage, and robustness issues
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.ops as ops
import logging

logger = logging.getLogger(__name__)

# Modern weights API imports for different model types
try:
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights
    MODERN_WEIGHTS_API = True
except ImportError:
    # Fallback for older torchvision versions
    MODERN_WEIGHTS_API = False
    logger.warning("Using legacy torchvision API. Consider upgrading to torchvision >= 0.13.0")


class FloorPlanFasterRCNN(nn.Module):
    """
    Faster R-CNN implementation for floor plan object detection
    
    Features:
    - Thread-safe prediction with manual post-processing 
    - Modern torchvision weights API support
    - Robust backbone freezing logic
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Faster R-CNN model
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config

        # Model parameters
        self.num_classes = config.get('num_classes', 4)  # Background + 3 classes
        self.backbone_name = config.get('backbone', 'resnet50')
        self.pretrained = config.get('pretrained', True)
        self.trainable_backbone_layers = config.get('trainable_backbone_layers', 3)

        # Post-processing parameters
        self.default_score_thresh = config.get('box_score_thresh', 0.05)
        self.default_nms_thresh = config.get('box_nms_thresh', 0.5)
        self.max_detections_per_img = config.get('box_detections_per_img', 100)

        # Build model
        self._build_model()

        # Apply backbone freezing
        self._apply_backbone_freezing()
        
        logger.info(f"Initialized FloorPlanFasterRCNN: {self.num_classes} classes, "
                   f"backbone: {self.backbone_name}, pretrained: {self.pretrained}")
    
    def _build_model(self):
        """Build the Faster R-CNN model"""
        
        model_name = self.config.get('model_name', 'fasterrcnn_resnet50_fpn')
        
        if 'resnet50' in model_name.lower():
            self.model = self._create_resnet50_model()
        elif 'mobilenet' in model_name.lower():
            self.model = self._create_mobilenet_model()
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}")
        
        # Replace classifier head if needed
        if self.num_classes != 91:  # COCO has 91 classes including background
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
    
    def _create_resnet50_model(self) -> FasterRCNN:
        """Create ResNet50-based Faster R-CNN"""
        
        if MODERN_WEIGHTS_API and self.pretrained:
            # Use modern weights API
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = fasterrcnn_resnet50_fpn(
                weights=weights,
                rpn_score_thresh=self.config.get('rpn_score_thresh', 0.0),
                box_score_thresh=self.default_score_thresh,
                box_nms_thresh=self.default_nms_thresh,
                box_detections_per_img=self.max_detections_per_img
            )
        else:
            # Fallback to legacy API
            model = fasterrcnn_resnet50_fpn(
                pretrained=self.pretrained,
                num_classes=91 if self.pretrained else self.num_classes,
                rpn_score_thresh=self.config.get('rpn_score_thresh', 0.0),
                box_score_thresh=self.default_score_thresh,
                box_nms_thresh=self.default_nms_thresh,
                box_detections_per_img=self.max_detections_per_img
            )
        
        return model
    
    def _create_mobilenet_model(self) -> FasterRCNN:
        """Create MobileNet-based Faster R-CNN"""
        
        if MODERN_WEIGHTS_API and self.pretrained:
            # Use modern weights API 
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            model = fasterrcnn_mobilenet_v3_large_fpn(
                weights=weights,
                rpn_score_thresh=self.config.get('rpn_score_thresh', 0.0),
                box_score_thresh=self.default_score_thresh,
                box_nms_thresh=self.default_nms_thresh,
                box_detections_per_img=self.max_detections_per_img
            )
        else:
            # Fallback to legacy API
            model = fasterrcnn_mobilenet_v3_large_fpn(
                pretrained=self.pretrained,
                num_classes=91 if self.pretrained else self.num_classes,
                rpn_score_thresh=self.config.get('rpn_score_thresh', 0.0),
                box_score_thresh=self.default_score_thresh,
                box_nms_thresh=self.default_nms_thresh,
                box_detections_per_img=self.max_detections_per_img
            )
        
        return model
    
    def _apply_backbone_freezing(self):
        """
        Apply robust backbone freezing based on layer indices
        """
        if not hasattr(self.model, 'backbone') or not hasattr(self.model.backbone, 'body'):
            logger.warning("Cannot access backbone.body, skipping freezing")
            return
        
        backbone_body = self.model.backbone.body
        
        # Get all backbone layers as a list
        if 'resnet' in self.backbone_name.lower():
            self._freeze_resnet_backbone(backbone_body)
        elif 'mobilenet' in self.backbone_name.lower():
            self._freeze_mobilenet_backbone(backbone_body)
        else:
            logger.warning(f"Unknown backbone type: {self.backbone_name}, applying default freezing")
            self._freeze_default_backbone(backbone_body)
        
        # Log freezing statistics
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"Backbone freezing applied: {self.trainable_backbone_layers} trainable layers")
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                   f"({100 * trainable_params / total_params:.1f}%)")
    
    def _freeze_resnet_backbone(self, backbone_body):
        """Freeze ResNet backbone layers"""
        
        # ResNet layer hierarchy: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4]
        # We typically want to train the last few layers
        
        # First freeze everything
        for param in backbone_body.parameters():
            param.requires_grad = False
        
        # Get all named children and convert to list for index-based access
        all_layers = list(backbone_body.named_children())
        
        # Determine which layers to unfreeze based on trainable_backbone_layers
        # Layer4 = most important, then layer3, layer2, layer1, then early layers
        layers_priority = ['layer4', 'layer3', 'layer2', 'layer1', 'bn1', 'conv1']
        layers_to_unfreeze = layers_priority[:self.trainable_backbone_layers]
        
        for layer_name in layers_to_unfreeze:
            if hasattr(backbone_body, layer_name):
                layer = getattr(backbone_body, layer_name)
                for param in layer.parameters():
                    param.requires_grad = True
                logger.debug(f"Unfroze ResNet layer: {layer_name}")
    
    def _freeze_mobilenet_backbone(self, backbone_body):
        """Freeze MobileNet backbone layers - FIXED VERSION"""
        
        # First freeze everything
        for param in backbone_body.parameters():
            param.requires_grad = False
        
        # Find MobileNet features through various possible paths
        feature_layers = None
        features_path = ""
        
        # Method 1: Direct features access
        if hasattr(backbone_body, 'features'):
            feature_layers = backbone_body.features
            features_path = "backbone_body.features"
            
        # Method 2: Through wrapped model (IntermediateLayerGetter)  
        elif hasattr(backbone_body, 'model') and hasattr(backbone_body.model, 'features'):
            feature_layers = backbone_body.model.features
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
            
            feature_layers, features_path = find_features(backbone_body)
        
        if feature_layers is not None:
            # Unfreeze the last N feature layers
            total_layers = len(feature_layers)
            start_idx = max(0, total_layers - self.trainable_backbone_layers)
            
            unfrozen_params = 0
            for i in range(start_idx, total_layers):
                for param in feature_layers[i].parameters():
                    param.requires_grad = True
                    unfrozen_params += param.numel()
            
            logger.info(f"Successfully unfroze MobileNet layers {start_idx} to {total_layers-1} from {features_path}")
            logger.info(f"Unfrozen {unfrozen_params:,} backbone parameters")
        else:
            # Enhanced fallback with more detailed structure exploration
            logger.warning("Could not identify MobileNet features structure")
            logger.warning("Available backbone_body attributes:")
            for attr_name in dir(backbone_body):
                if not attr_name.startswith('_') and hasattr(getattr(backbone_body, attr_name), 'parameters'):
                    try:
                        attr = getattr(backbone_body, attr_name)
                        param_count = sum(p.numel() for p in attr.parameters())
                        logger.warning(f"  - {attr_name}: {type(attr).__name__} ({param_count:,} params)")
                    except:
                        pass
            
            # Fallback: unfreeze all if we can't identify structure
            for param in backbone_body.parameters():
                param.requires_grad = True
            logger.warning("Applied fallback: unfroze all backbone parameters")
    
    def _freeze_default_backbone(self, backbone_body):
        """Default freezing strategy"""
        
        # Get all modules and unfreeze the last N layers
        all_modules = list(backbone_body.modules())
        if len(all_modules) > self.trainable_backbone_layers:
            # Freeze all first
            for param in backbone_body.parameters():
                param.requires_grad = False
            
            # Unfreeze last layers
            for module in all_modules[-self.trainable_backbone_layers:]:
                for param in module.parameters():
                    param.requires_grad = True
        
        logger.warning("Applied default backbone freezing strategy")
    
    def forward(self, images: List[torch.Tensor], 
                targets: Optional[List[Dict[str, torch.Tensor]]] = None) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Forward pass
        
        Args:
            images: List of input images
            targets: Training targets (optional)
        
        Returns:
            Training losses or raw predictions
        """
        return self.model(images, targets)
    
    def predict(self, images: List[torch.Tensor], 
                score_threshold: float = None,
                nms_threshold: float = None,
                max_detections: int = None) -> List[Dict[str, torch.Tensor]]:
        """
        Thread-safe prediction with manual post-processing
        
        Args:
            images: Input images
            score_threshold: Score threshold for filtering
            nms_threshold: NMS threshold
            max_detections: Maximum detections per image
        
        Returns:
            Filtered predictions
        """
        # Use defaults if not specified  
        if score_threshold is None:
            score_threshold = self.default_score_thresh
        if nms_threshold is None:
            nms_threshold = self.default_nms_thresh
        if max_detections is None:
            max_detections = self.max_detections_per_img
        
        self.model.eval()
        
        with torch.no_grad():
            # Get raw predictions WITHOUT modifying model state
            raw_predictions = self.model(images)
        
        # Apply manual post-processing to each prediction
        filtered_predictions = []
        for pred in raw_predictions:
            filtered_pred = self._apply_postprocessing(
                pred, score_threshold, nms_threshold, max_detections
            )
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions
    
    def _apply_postprocessing(self, prediction: Dict[str, torch.Tensor], 
                            score_threshold: float,
                            nms_threshold: float, 
                            max_detections: int) -> Dict[str, torch.Tensor]:
        """
        Apply manual post-processing (scoring + NMS)
        
        Args:
            prediction: Raw model prediction
            score_threshold: Score threshold
            nms_threshold: NMS threshold
            max_detections: Maximum detections

        Returns:
            Filtered prediction
        """
        boxes = prediction['boxes']
        scores = prediction['scores']
        labels = prediction['labels']
        
        # Step 1: Filter by score threshold
        score_mask = scores > score_threshold
        boxes = boxes[score_mask]
        scores = scores[score_mask]
        labels = labels[score_mask]
        
        if len(boxes) == 0:
            # Return empty prediction
            return {
                'boxes': torch.empty((0, 4), device=boxes.device),
                'scores': torch.empty(0, device=scores.device),
                'labels': torch.empty(0, dtype=torch.int64, device=labels.device)
            }
        
        # Step 2: Apply NMS per class
        keep_indices = []
        unique_labels = labels.unique()
        
        for label in unique_labels:
            label_mask = labels == label
            label_boxes = boxes[label_mask]
            label_scores = scores[label_mask]
            
            # Apply NMS for this class
            nms_indices = ops.nms(label_boxes, label_scores, nms_threshold)
            
            # Convert back to global indices
            global_indices = torch.nonzero(label_mask).squeeze(1)[nms_indices]
            keep_indices.append(global_indices)
        
        if keep_indices:
            keep_indices = torch.cat(keep_indices)
            
            # Step 3: Sort by score and limit detections
            scores_after_nms = scores[keep_indices]
            _, sorted_indices = scores_after_nms.sort(descending=True)
            
            # Limit to max detections
            if len(sorted_indices) > max_detections:
                sorted_indices = sorted_indices[:max_detections]
            
            final_indices = keep_indices[sorted_indices]
            
            return {
                'boxes': boxes[final_indices],
                'scores': scores[final_indices], 
                'labels': labels[final_indices]
            }
        else:
            # No detections after NMS
            return {
                'boxes': torch.empty((0, 4), device=boxes.device),
                'scores': torch.empty(0, device=scores.device),
                'labels': torch.empty(0, dtype=torch.int64, device=labels.device)
            }
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters for logging"""
        return {
            'num_classes': self.num_classes,
            'backbone': self.backbone_name,
            'pretrained': self.pretrained,
            'trainable_backbone_layers': self.trainable_backbone_layers,
            'default_score_thresh': self.default_score_thresh,
            'default_nms_thresh': self.default_nms_thresh,
            'max_detections_per_img': self.max_detections_per_img,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 * 1024)
        }


def create_faster_rcnn_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create Faster R-CNN model
    
    Args:
        config: Configuration dictionary

    Returns:
        Faster R-CNN model
    """
    return FloorPlanFasterRCNN(config)


def load_faster_rcnn_checkpoint(model: nn.Module, checkpoint_path: str, 
                               strict: bool = True) -> nn.Module:
    """
    Load model from checkpoint with error handling
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint
        strict: Whether to strictly enforce state dict matching
        
    Returns:
        Model with loaded weights
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Handle potential key mismatches
        if hasattr(model, 'model'):
            # If our model wraps the actual Faster R-CNN
            try:
                model.model.load_state_dict(state_dict, strict=strict)
            except RuntimeError as e:
                if not strict:
                    logger.warning(f"Some keys didn't match when loading checkpoint: {e}")
                    model.model.load_state_dict(state_dict, strict=False)
                else:
                    raise
        else:
            model.load_state_dict(state_dict, strict=strict)
        
        logger.info(f"Successfully loaded Faster R-CNN checkpoint from {checkpoint_path}")
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        raise
    
    return model


# For backward compatibility
SimpleFasterRCNN = FloorPlanFasterRCNN


# Utility functions
def count_model_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


def get_model_device(model: nn.Module) -> torch.device:
    """Get the device of the model"""
    return next(model.parameters()).device