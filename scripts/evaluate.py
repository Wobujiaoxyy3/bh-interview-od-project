"""
Evaluation script for floor plan object detection models
Evaluates trained models on test data with comprehensive metrics
"""

import sys
import argparse
import logging
from pathlib import Path
import yaml
import json
import torch
from typing import Dict, List, Tuple, Any, Optional, Union

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ConfigManager
from models import create_faster_rcnn_model, create_retinanet_model
from datasets import create_datasets
from evaluation import COCOEvaluator, create_evaluator
from utils import setup_logging

# Import type definitions 
from custom_types import (
    Device, COCOPrediction, EvaluationMetrics, ModelPrediction,
    DatasetSample, ModelTarget
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate floor plan object detection models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and configuration
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Data
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    
    # Evaluation parameters
    parser.add_argument('--confidence-threshold', type=float, default=0.05,
                       help='Confidence threshold for predictions')
    parser.add_argument('--nms-threshold', type=float, default=0.5,
                       help='NMS threshold for predictions')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Evaluation batch size')
    
    # Output
    parser.add_argument('--output-dir', '-o', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save predictions in COCO format')
    
    # Hardware
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser.parse_args()


def load_model(config: Dict[str, Any], checkpoint_path: Union[str, Path], device: Device) -> torch.nn.Module:
    """Load model from checkpoint"""
    logger = logging.getLogger(__name__)
    
    model_config = config.get('model', {})
    model_type = model_config.get('architecture', 'faster_rcnn')
    
    # Create model
    if model_type == 'faster_rcnn':
        model = create_faster_rcnn_model(model_config)
    elif model_type == 'retinanet':
        model = create_retinanet_model(model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded {model_type} model from {checkpoint_path}")
    
    return model


def evaluate_model(model: torch.nn.Module, 
                  data_loader: torch.utils.data.DataLoader[Any], 
                  evaluator: COCOEvaluator, 
                  device: Device,
                  confidence_threshold: float = 0.05, 
                  nms_threshold: float = 0.5, 
                  save_predictions: bool = False, 
                  output_dir: Optional[Union[str, Path]] = None) -> Tuple[EvaluationMetrics, List[COCOPrediction]]:
    """Evaluate model on dataset"""
    logger = logging.getLogger(__name__)
    logger.info("Starting model evaluation...")
    
    predictions: List[COCOPrediction] = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if batch_idx % 50 == 0:
                logger.info(f"Processing batch {batch_idx}/{len(data_loader)}")
            
            # Move images to device
            images = [img.to(device) for img in images]
            
            # Get model predictions
            outputs = model(images)
            
            # Convert predictions from Pascal VOC to COCO format
            for i, output in enumerate(outputs):
                image_id = targets[i]['image_id'].item()
                
                if len(output['boxes']) == 0:
                    continue
                
                boxes = output['boxes'].cpu().numpy()  # Pascal VOC format (x1, y1, x2, y2)
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                
                # Filter by confidence threshold
                valid_indices = scores >= confidence_threshold
                boxes = boxes[valid_indices]
                scores = scores[valid_indices]
                labels = labels[valid_indices]
                
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    # Convert Pascal VOC (x1, y1, x2, y2) to COCO (x, y, width, height)
                    width = x2 - x1
                    height = y2 - y1
                    predictions.append({
                        'image_id': int(image_id),
                        'category_id': int(label),
                        'bbox': [float(x1), float(y1), float(width), float(height)],  # COCO format
                        'score': float(score)
                    })
    
    logger.info(f"Generated {len(predictions)} predictions")
    
    # Save predictions if requested
    if save_predictions and output_dir:
        predictions_file = Path(output_dir) / 'predictions.json'
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"Predictions saved to {predictions_file}")
    
    # Evaluate predictions
    metrics = evaluator.evaluate(
        predictions, 
        confidence_threshold=0.0,  # Already filtered
        nms_threshold=nms_threshold
    )
    
    return metrics, predictions


def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model evaluation")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Split: {args.split}")
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        # Setup device
        if args.device:
            device = args.device
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        device = torch.device(device)
        logger.info(f"Using device: {device}")
        
        # Override batch size if specified
        if args.batch_size:
            config.setdefault('data_loader', {})['batch_size'] = args.batch_size
        
        # Override num workers
        config.setdefault('data_loader', {})['num_workers'] = args.num_workers
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Load model
        model = load_model(config, args.checkpoint, device)
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = create_datasets(config)
        
        # Select dataset split
        if args.split == 'train':
            eval_dataset = train_dataset
        elif args.split == 'val':
            eval_dataset = val_dataset
        else:
            eval_dataset = test_dataset
        
        # Create data loader
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=config.get('data_loader', {}).get('batch_size', 4),
            shuffle=False,
            num_workers=config.get('data_loader', {}).get('num_workers', 4),
            pin_memory=True,
            collate_fn=train_dataset.__class__.collate_fn if hasattr(train_dataset.__class__, 'collate_fn') else None
        )
        
        # Create evaluator
        evaluator = COCOEvaluator(
            str(eval_dataset.annotations_file),
            device=device
        )
        
        # Run evaluation
        metrics, predictions = evaluate_model(
            model=model,
            data_loader=eval_loader,
            evaluator=evaluator,
            device=device,
            confidence_threshold=args.confidence_threshold,
            nms_threshold=args.nms_threshold,
            save_predictions=args.save_predictions,
            output_dir=output_dir
        )
        
        # Print results
        evaluator.print_metrics(metrics)
        
        # Save results
        results_file = output_dir / 'evaluation_results.json'
        results = {
            'checkpoint': str(args.checkpoint),
            'split': args.split,
            'confidence_threshold': args.confidence_threshold,
            'nms_threshold': args.nms_threshold,
            'num_predictions': len(predictions),
            'metrics': metrics
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_file}")
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()