"""
COCO evaluation metrics for object detection
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict
import tempfile

# Import type definitions
try:
    from ..custom_types import (Device, COCOPrediction, EvaluationMetrics)

except ImportError:
    # Handle direct script execution
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from custom_types import (Device, COCOPrediction, EvaluationMetrics)

logger = logging.getLogger(__name__)


class COCOEvaluator:
    """
    Fixed COCO-style evaluator for object detection models

    Key fixes implemented:
    - Removed redundant NMS handling (delegates to model and pycocotools)
    - Simplified confidence threshold handling
    - Proper per-category AP calculation using pycocotools results
    """
    
    # Type annotations for instance variables
    annotations_file: Path
    device: Device
    coco_gt: COCO
    category_ids: List[int]
    category_names: Dict[int, str]
    iou_thresholds: List[float]
    area_ranges: List[Tuple[int, int]]
    
    def __init__(self, 
                 annotations_file: Union[str, Path],
                 device: Optional[Device] = None,
                 iou_thresholds: Optional[List[float]] = None,
                 area_ranges: Optional[List[Tuple[int, int]]] = None) -> None:
        """
        Initialize COCO evaluator
        
        Args:
            annotations_file: Path to COCO format annotations
            device: Device for computation
            iou_thresholds: List of IoU thresholds for evaluation (uses COCO default if None)
            area_ranges: List of (min_area, max_area) tuples for different object sizes
        """
        self.annotations_file = Path(annotations_file)
        self.device = device or torch.device('cpu')
        
        # Load ground truth annotations
        if self.annotations_file.exists():
            self.coco_gt = COCO(str(self.annotations_file))
            self.category_ids = list(self.coco_gt.getCatIds())
            self.category_names = {cat['id']: cat['name'] for cat in self.coco_gt.loadCats(self.category_ids)}
        else:
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
        
        # Evaluation parameters - use COCO defaults
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.area_ranges = area_ranges or [
            (0, 10000000000),  # all
            (0, 32*32),        # small
            (32*32, 96*96),    # medium
            (96*96, 10000000000)  # large
        ]
        
        logger.info(f"Initialized COCOEvaluator with {len(self.category_ids)} categories: {list(self.category_names.values())}")
    
    def evaluate(self, predictions: List[COCOPrediction], 
                 confidence_threshold: float = 0.0) -> EvaluationMetrics:
        """
        Evaluate predictions using COCO metrics

        Args:
            predictions: List of prediction dictionaries in COCO format
            confidence_threshold: Minimum confidence threshold (applied only if > 0.0)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not predictions:
            logger.warning("No predictions provided for evaluation")
            return self._get_empty_metrics()
        
        # FIX: Minimal confidence filtering, no NMS
        filtered_predictions = predictions
        if confidence_threshold > 0.0:
            filtered_predictions = [
                pred for pred in predictions 
                if pred.get('score', 0) >= confidence_threshold
            ]
            logger.info(f"Filtered predictions: {len(predictions)} -> {len(filtered_predictions)} (threshold: {confidence_threshold})")
        
        if not filtered_predictions:
            logger.warning("No predictions above confidence threshold")
            return self._get_empty_metrics()
        
        # FIX: Let pycocotools handle everything else
        return self._evaluate_with_pycocotools(filtered_predictions)
    
    def _evaluate_with_pycocotools(self, predictions: List[COCOPrediction]) -> EvaluationMetrics:
        """
        Core evaluation using pycocotools - no manual NMS or filtering
        
        Args:
            predictions: List of predictions in COCO format
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Validate prediction format
        self._validate_predictions(predictions)
        
        # Create temporary file for predictions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(predictions, f, indent=2)
            predictions_file = f.name
        
        try:
            # Load predictions and run evaluation
            logger.debug(f"Loading {len(predictions)} predictions from temporary file")
            coco_dt = self.coco_gt.loadRes(predictions_file)
            coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
            
            # Configure evaluation parameters
            coco_eval.params.iouThrs = np.array(self.iou_thresholds)
            coco_eval.params.areaRng = [[ar[0], ar[1]] for ar in self.area_ranges]
            coco_eval.params.areaRngLbl = ['all', 'small', 'medium', 'large'][:len(self.area_ranges)]
            
            # Run evaluation
            logger.debug("Running COCO evaluation...")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract metrics
            metrics = self._extract_metrics_from_coco(coco_eval)
            
            logger.info(f"Evaluation completed: mAP={metrics.get('mAP', 0):.4f}, mAP@0.5={metrics.get('mAP_50', 0):.4f}")
            
        except Exception as e:
            logger.error(f"COCO evaluation failed: {e}")
            return self._get_empty_metrics()
        finally:
            # Clean up temporary file
            Path(predictions_file).unlink(missing_ok=True)
        
        return metrics
    
    def _validate_predictions(self, predictions: List[COCOPrediction]) -> None:
        """
        Validate prediction format
        """
        if not predictions:
            return
        
        sample_pred = predictions[0]
        required_keys = ['image_id', 'category_id', 'bbox', 'score']
        
        for key in required_keys:
            if key not in sample_pred:
                raise ValueError(f"Missing required key '{key}' in prediction")
        
        # Validate bbox format (should be COCO: [x, y, width, height])
        bbox = sample_pred['bbox']
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"Invalid bbox format: {bbox}. Expected [x, y, width, height]")
        
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            logger.warning(f"Found bbox with invalid dimensions: w={w}, h={h}")
    
    def _extract_metrics_from_coco(self, coco_eval: COCOeval) -> Dict[str, float]:
        """
        Extract evaluation metrics from COCOeval results
        
        Args:
            coco_eval: COCOeval object with computed results
            
        Returns:
            Dictionary of metrics
        """
        stats = coco_eval.stats
        
        # Use standard COCO metrics extraction
        metrics = {
            'mAP': float(stats[0]) if len(stats) > 0 else 0.0,           # AP@[0.5:0.95]
            'mAP_50': float(stats[1]) if len(stats) > 1 else 0.0,        # AP@0.5
            'mAP_75': float(stats[2]) if len(stats) > 2 else 0.0,        # AP@0.75
            'mAP_small': float(stats[3]) if len(stats) > 3 else 0.0,     # AP@[0.5:0.95] small
            'mAP_medium': float(stats[4]) if len(stats) > 4 else 0.0,    # AP@[0.5:0.95] medium
            'mAP_large': float(stats[5]) if len(stats) > 5 else 0.0,     # AP@[0.5:0.95] large
            'mAR_1': float(stats[6]) if len(stats) > 6 else 0.0,         # AR@[0.5:0.95] maxDets=1
            'mAR_10': float(stats[7]) if len(stats) > 7 else 0.0,        # AR@[0.5:0.95] maxDets=10
            'mAR_100': float(stats[8]) if len(stats) > 8 else 0.0,       # AR@[0.5:0.95] maxDets=100
            'mAR_small': float(stats[9]) if len(stats) > 9 else 0.0,     # AR@[0.5:0.95] small
            'mAR_medium': float(stats[10]) if len(stats) > 10 else 0.0,  # AR@[0.5:0.95] medium
            'mAR_large': float(stats[11]) if len(stats) > 11 else 0.0    # AR@[0.5:0.95] large
        }
        
        # Proper per-category metrics using pycocotools results
        per_category_metrics = self._extract_per_category_metrics_from_coco(coco_eval)
        metrics.update(per_category_metrics)
        
        return metrics
    
    def _extract_per_category_metrics_from_coco(self, coco_eval: COCOeval) -> Dict[str, float]:
        """
        Extract per-category AP metrics using proper pycocotools method
        
        Args:
            coco_eval: COCOeval object with results
            
        Returns:
            Dictionary of per-category metrics
        """
        per_category_metrics = {}
        
        try:
            # Use proper COCO evaluation results
            # Get per-category results from the evaluation
            # eval['precision'] has shape (T, R, K, A, M) where:
            # T = IoU thresholds, R = recall thresholds, K = categories, A = area ranges, M = max detections
            if coco_eval.eval is None:
                logger.warning("No evaluation results available for per-category metrics")
                return per_category_metrics
            
            precision_scores = coco_eval.eval['precision']  # (T, R, K, A, M)
            
            # For each category, compute AP@[0.5:0.95] and AP@0.5
            for cat_idx, cat_id in enumerate(self.category_ids):
                if cat_idx >= precision_scores.shape[2]:
                    continue
                
                cat_name = self.category_names.get(cat_id, f'category_{cat_id}')
                
                # AP@[0.5:0.95] - average over all IoU thresholds (area=all, maxDets=100)
                # Shape: (T=10, R=101) -> average over valid entries
                cat_precision_all_iou = precision_scores[:, :, cat_idx, 0, -1]  # All IoU, all recalls, area=all, maxDets=100
                valid_precision = cat_precision_all_iou[cat_precision_all_iou > -1]
                cat_ap = np.mean(valid_precision) if len(valid_precision) > 0 else 0.0
                
                # AP@0.5 - only IoU threshold 0.5 (area=all, maxDets=100)
                cat_precision_50 = precision_scores[0, :, cat_idx, 0, -1]  # IoU=0.5, all recalls
                valid_precision_50 = cat_precision_50[cat_precision_50 > -1]
                cat_ap_50 = np.mean(valid_precision_50) if len(valid_precision_50) > 0 else 0.0
                
                per_category_metrics[f'AP_{cat_name}'] = float(cat_ap)
                per_category_metrics[f'AP50_{cat_name}'] = float(cat_ap_50)
                
                logger.debug(f"Category {cat_name}: AP={cat_ap:.4f}, AP@0.5={cat_ap_50:.4f}")
            
        except Exception as e:
            logger.warning(f"Failed to compute per-category metrics: {e}")
        
        return per_category_metrics
    
    def _get_empty_metrics(self) -> Dict[str, float]:
        """Get empty metrics dictionary with all values set to 0"""
        metrics = {
            'mAP': 0.0,
            'mAP_50': 0.0,
            'mAP_75': 0.0,
            'mAP_small': 0.0,
            'mAP_medium': 0.0,
            'mAP_large': 0.0,
            'mAR_1': 0.0,
            'mAR_10': 0.0,
            'mAR_100': 0.0,
            'mAR_small': 0.0,
            'mAR_medium': 0.0,
            'mAR_large': 0.0
        }
        
        # Add per-category empty metrics
        for cat_id in self.category_ids:
            cat_name = self.category_names.get(cat_id, f'category_{cat_id}')
            metrics[f'AP_{cat_name}'] = 0.0
            metrics[f'AP50_{cat_name}'] = 0.0
        
        return metrics
    
    def evaluate_model(self, model: torch.nn.Module, 
                      data_loader: torch.utils.data.DataLoader[Any],
                      confidence_threshold: float = 0.0) -> EvaluationMetrics:
        """
        Evaluate a model on a dataset
        
        Args:
            model: PyTorch model in eval mode
            data_loader: Data loader for evaluation
            confidence_threshold: Minimum confidence threshold (only applied if > 0.0)
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        predictions = []
        
        logger.info(f"Starting model evaluation with {len(data_loader)} batches")
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(data_loader):
                # Move images to device
                images = [img.to(self.device) for img in images]
                
                # FIX: Get model predictions WITHOUT additional filtering
                # The model in eval() mode already applies its internal thresholds
                outputs = model(images)
                
                # Convert predictions from Pascal VOC to COCO format
                for i, output in enumerate(outputs):
                    if len(output['boxes']) == 0:
                        continue
                    
                    image_id = targets[i]['image_id'].item()
                    boxes = output['boxes'].cpu().numpy()  # Pascal VOC format (x1, y1, x2, y2)
                    scores = output['scores'].cpu().numpy()
                    labels = output['labels'].cpu().numpy()
                    
                    # FIX: No additional confidence filtering here
                    # Model already filtered based on its box_score_thresh
                    for box, score, label in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = box
                        # Convert Pascal VOC (x1, y1, x2, y2) to COCO (x, y, width, height)
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Additional sanity check for valid boxes
                        if width > 0 and height > 0:
                            predictions.append({
                                'image_id': int(image_id),
                                'category_id': int(label),
                                'bbox': [float(x1), float(y1), float(width), float(height)],  # COCO format
                                'score': float(score)
                            })
                
                if batch_idx % 10 == 0:
                    logger.debug(f"Processed {batch_idx+1}/{len(data_loader)} batches, {len(predictions)} predictions so far")
        
        logger.info(f"Collected {len(predictions)} predictions from model")
        
        # FIX: Evaluate without additional NMS
        # pycocotools handles the rest
        return self.evaluate(predictions, confidence_threshold=confidence_threshold)
    
    def print_metrics(self, metrics: Dict[str, float]):
        """
        Print evaluation metrics in a readable format
        
        Args:
            metrics: Dictionary of metrics from evaluate()
        """
        logger.info("=" * 60)
        logger.info("COCO Evaluation Results")
        logger.info("=" * 60)
        
        # Main metrics
        logger.info(f"Average Precision (AP):")
        logger.info(f"  AP@[0.5:0.95] = {metrics.get('mAP', 0):.4f}")
        logger.info(f"  AP@0.5       = {metrics.get('mAP_50', 0):.4f}")
        logger.info(f"  AP@0.75      = {metrics.get('mAP_75', 0):.4f}")
        
        # Size-based metrics
        logger.info(f"  AP (small)   = {metrics.get('mAP_small', 0):.4f}")
        logger.info(f"  AP (medium)  = {metrics.get('mAP_medium', 0):.4f}")
        logger.info(f"  AP (large)   = {metrics.get('mAP_large', 0):.4f}")
        
        # Recall metrics
        logger.info(f"Average Recall (AR):")
        logger.info(f"  AR@1         = {metrics.get('mAR_1', 0):.4f}")
        logger.info(f"  AR@10        = {metrics.get('mAR_10', 0):.4f}")
        logger.info(f"  AR@100       = {metrics.get('mAR_100', 0):.4f}")
        
        # Per-category metrics
        logger.info(f"Per-category AP@[0.5:0.95]:")
        for cat_id in self.category_ids:
            cat_name = self.category_names.get(cat_id, f'category_{cat_id}')
            ap_key = f'AP_{cat_name}'
            ap50_key = f'AP50_{cat_name}'
            if ap_key in metrics:
                logger.info(f"  {cat_name:10s} = {metrics[ap_key]:.4f} (AP@0.5: {metrics.get(ap50_key, 0):.4f})")
        
        logger.info("=" * 60)


def create_evaluator(config: Dict[str, Any]) -> COCOEvaluator:
    """
    Factory function to create COCO evaluator from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        COCOEvaluator instance
    """
    eval_config = config.get('evaluation', {})
    data_config = config.get('data', {})
    
    # Get validation annotations file
    splits_dir = Path(data_config.get('splits_dir', 'data/processed/splits'))
    annotations_file = splits_dir / 'val_annotations.json'
    
    if not annotations_file.exists():
        raise FileNotFoundError(f"Validation annotations not found: {annotations_file}")
    
    evaluator = COCOEvaluator(
        annotations_file=str(annotations_file),
        device=torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')),
        iou_thresholds=eval_config.get('iou_thresholds'),
        area_ranges=eval_config.get('area_ranges')
    )
    
    return evaluator