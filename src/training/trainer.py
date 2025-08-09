"""
Training pipeline for floor plan object detection models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from pathlib import Path
import time
import json
import numpy as np
from collections import defaultdict

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

try:
    import wandb
except ImportError:
    wandb = None

try:
    from ..models import create_faster_rcnn_model, create_retinanet_model
    from ..datasets import create_data_loaders
    from ..evaluation import COCOEvaluator
except ImportError:
    # Handle direct script execution
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from models import create_faster_rcnn_model, create_retinanet_model
    from datasets import create_data_loaders
    from evaluation import COCOEvaluator

logger = logging.getLogger(__name__)


class ObjectDetectionTrainer:
    """
    Fixed trainer class for object detection models
    
    Key fixes implemented:
    - Proper validation mode (model.eval() instead of model.train())
    - Correct checkpoint loading order 
    - Simplified COCO evaluation without redundant filtering
    """
    
    # Type annotations for instance variables
    config: Dict[str, Any]
    device: Device
    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: Optional[optim.lr_scheduler._LRScheduler]
    train_loader: DataLoader[Any]
    val_loader: DataLoader[Any] 
    test_loader: DataLoader[Any]
    evaluator: Optional['COCOEvaluator']
    current_epoch: int
    global_step: int
    best_map: float
    training_history: Dict[str, List[float]]
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize trainer with fixed checkpoint loading order
        
        Args:
            config: Complete configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Training parameters
        self.num_epochs = config.get('training', {}).get('num_epochs', 50)
        self.save_every = config.get('training', {}).get('save_every', 5)
        self.validate_every = config.get('training', {}).get('validate_every', 1)
        self.log_every = config.get('training', {}).get('log_every', 10)
        
        # Get config name for subfolder creation
        self.config_name = config.get('_meta', {}).get('config_name', 'default_config')
        
        # Create config-specific subdirectory structure
        base_output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir = base_output_dir / self.config_name
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Save config file to this training session's directory
        self._save_config()
        
        # Initialize components in correct order
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.train_loader, self.val_loader, self.test_loader = self._create_data_loaders()
        
        # Initialize evaluator
        self.evaluator = self._create_evaluator()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_map = 0.0
        self.training_history = defaultdict(list)
        
        # Load checkpoint AFTER all components are created
        checkpoint_path = self.config.get('model', {}).get('checkpoint_path')
        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_checkpoint(checkpoint_path)
        
        # Initialize wandb if configured
        if config.get('use_wandb', False) and wandb is not None:
            self._init_wandb()
        
        logger.info(f"Initialized trainer with {self.model.__class__.__name__} on {self.device}")
    
    def _save_config(self) -> None:
        """Save the configuration file used for this training session"""
        import yaml
        from datetime import datetime
        
        # Save the complete config
        config_save_path = self.output_dir / f'{self.config_name}_config.yaml'
        
        # Create a clean copy of config without metadata for saving
        config_to_save = {}
        for key, value in self.config.items():
            if key != '_meta':  # Exclude metadata from saved config
                config_to_save[key] = value
        
        # Add training session metadata
        config_to_save['training_session'] = {
            'config_name': self.config_name,
            'created_at': datetime.now().isoformat(),
            'output_directory': str(self.output_dir),
            'device': str(self.device)
        }
        
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, indent=2, allow_unicode=True)
        
        logger.info(f"Saved configuration to: {config_save_path}")
        logger.info(f"Training outputs will be saved to: {self.output_dir}")
        logger.info(f"  - Checkpoints: {self.checkpoint_dir}")
        logger.info(f"  - Logs: {self.log_dir}")
    
    def _save_training_history(self) -> None:
        """Save training history incrementally after each epoch"""
        history_path = self.log_dir / 'training_history.json'
        
        try:
            # Convert defaultdict to regular dict for JSON serialization
            history_dict = dict(self.training_history)
            
            # Add metadata about the current training state
            history_dict['_metadata'] = {
                'current_epoch': self.current_epoch,
                'global_step': self.global_step,
                'best_map': self.best_map,
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
                'config_name': self.config_name
            }
            
            # Save to JSON file
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save training history: {e}")
    
    def _create_model(self) -> nn.Module:
        """Create and initialize model without checkpoint loading"""
        model_config = self.config.get('model', {})
        model_type = model_config.get('architecture', 'faster_rcnn')
        
        if model_type == 'faster_rcnn':
            model = create_faster_rcnn_model(model_config)
        elif model_type == 'retinanet':
            model = create_retinanet_model(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model.to(self.device)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        optim_config = self.config.get('optimizer', {})
        optimizer_type = optim_config.get('type', 'adamw')
        learning_rate = float(optim_config.get('learning_rate', 1e-4))
        weight_decay = float(optim_config.get('weight_decay', 1e-4))
        
        if optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=optim_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=float(optim_config.get('momentum', 0.9)),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        scheduler_config = self.config.get('scheduler', {})
        if not scheduler_config.get('use_scheduler', False):
            return None
        
        scheduler_type = scheduler_config.get('type', 'step')
        
        if scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(scheduler_config.get('step_size', 10)),
                gamma=float(scheduler_config.get('gamma', 0.1))
            )
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=float(scheduler_config.get('eta_min', 1e-6))
            )
        elif scheduler_type == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=float(scheduler_config.get('factor', 0.5)),
                patience=int(scheduler_config.get('patience', 5)),
                min_lr=float(scheduler_config.get('min_lr', 1e-6))
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
        
        return scheduler
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders"""
        return create_data_loaders(self.config)
    
    def _create_evaluator(self) -> Optional['COCOEvaluator']:
        """Create COCO evaluator with proper configuration"""
        try:
            val_annotations_file = self.val_loader.dataset.annotations_file
            if val_annotations_file.exists():
                # Pass evaluation config to evaluator
                eval_config = self.config.get('evaluation', {})
                evaluator = COCOEvaluator(
                    annotations_file=str(val_annotations_file),
                    device=self.device
                )
                return evaluator
            else:
                logger.warning("Validation annotations not found. Evaluation disabled.")
                return None
        except Exception as e:
            logger.warning(f"Failed to create COCO evaluator: {e}")
            return None
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        if wandb is None:
            logger.warning("wandb not available, skipping initialization")
            return
            
        wandb_config = self.config.get('wandb', {})
        wandb.init(
            project=wandb_config.get('project', 'floor-plan-detection'),
            name=wandb_config.get('run_name', f"run_{int(time.time())}"),
            config=self.config,
            tags=wandb_config.get('tags', [])
        )
        wandb.watch(self.model, log='all')
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_losses = defaultdict(list)
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # Move data to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            
            # Compute total loss
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            losses.backward()
            
            # Gradient clipping
            if self.config.get('training', {}).get('gradient_clipping', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clipping']
                )
            
            self.optimizer.step()
            
            # Log losses
            for key, value in loss_dict.items():
                epoch_losses[key].append(value.item())
            epoch_losses['total_loss'].append(losses.item())
            
            # Log batch metrics
            if batch_idx % self.log_every == 0:
                logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                           f"Loss: {losses.item():.4f}")
                
                if wandb is not None and wandb.run:
                    wandb.log({
                        'train/batch_loss': losses.item(),
                        'train/step': self.global_step
                    })
            
            self.global_step += 1
        
        # Compute epoch averages
        epoch_metrics = {}
        for key, values in epoch_losses.items():
            epoch_metrics[f'train/{key}'] = np.mean(values)
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch

        
        Returns:
            Dictionary of validation metrics
        """

        self.model.eval()
        val_metrics = {}
        
        # Run COCO evaluation if evaluator is available
        if self.evaluator:
            try:
                coco_metrics = self.evaluate_coco()
                val_metrics.update(coco_metrics)
            except Exception as e:
                logger.warning(f"COCO evaluation failed: {e}")
                # Return empty metrics if evaluation fails
                val_metrics = {'val/coco_mAP': 0.0}
        else:
            logger.warning("No evaluator available, skipping validation metrics")
            val_metrics = {'val/coco_mAP': 0.0}
        
        return val_metrics
    
    def evaluate_coco(self) -> EvaluationMetrics:
        """
        Run COCO evaluation
        
        Returns:
            Dictionary of COCO metrics
        """
        if not self.evaluator:
            return {}
        
        # Ensure model is in eval mode
        self.model.eval()
        predictions: List[COCOPrediction] = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = [img.to(self.device) for img in images]
                
                # Get predictions (model handles thresholding internally)
                outputs = self.model(images)
                
                # Convert predictions from Pascal VOC to COCO format
                for i, output in enumerate(outputs):
                    image_id: int = targets[i]['image_id'].item()
                    boxes: np.ndarray = output['boxes'].cpu().numpy()  # Pascal VOC format in transformed coordinates
                    scores: np.ndarray = output['scores'].cpu().numpy()
                    labels: np.ndarray = output['labels'].cpu().numpy()
                    
                    # Get transformation parameters for inverse transform
                    scale = targets[i].get('scale', torch.tensor(1.0)).item()
                    pad_left = targets[i].get('pad_left', torch.tensor(0.0)).item()
                    pad_top = targets[i].get('pad_top', torch.tensor(0.0)).item()
                    
                    # Model in eval() mode already applies box_score_thresh filtering

                    for box, score, label in zip(boxes, scores, labels):
                        # Box is in transformed coordinates (800x800 padded image)
                        # Need to convert back to original image coordinates
                        x1, y1, x2, y2 = box
                        
                        # Inverse transform: remove padding and scale back
                        x1_orig = (x1 - pad_left) / scale
                        y1_orig = (y1 - pad_top) / scale
                        x2_orig = (x2 - pad_left) / scale
                        y2_orig = (y2 - pad_top) / scale
                        
                        # Convert Pascal VOC (x1, y1, x2, y2) to COCO (x, y, width, height)
                        width: float = x2_orig - x1_orig
                        height: float = y2_orig - y1_orig
                        
                        # Validate box dimensions
                        if width > 0 and height > 0:
                            # Create COCO prediction with original coordinates
                            coco_prediction: COCOPrediction = {
                                'image_id': image_id,
                                'category_id': int(label),
                                'bbox': (float(x1_orig), float(y1_orig), float(width), float(height)),
                                'score': float(score)
                            }
                            predictions.append(coco_prediction)
        
        # Evaluate predictions
        metrics = self.evaluator.evaluate(predictions)
        
        # Format metrics for logging
        formatted_metrics = {}
        for key, value in metrics.items():
            formatted_metrics[f'val/coco_{key}'] = value
        
        return formatted_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_map': self.best_map,
            'config': self.config,
            'training_history': dict(self.training_history),
            'global_step': self.global_step
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with mAP: {self.best_map:.4f}")
        
        # Save latest model
        latest_path = self.checkpoint_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_path)
        
        logger.info(f"Saved checkpoint at epoch {epoch}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                logger.warning("No model_state_dict found in checkpoint")
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint and self.optimizer:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    logger.warning(f"Failed to load optimizer state: {e}")
            
            # Load scheduler state
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    logger.warning(f"Failed to load scheduler state: {e}")
            
            # Load training state
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_map = checkpoint.get('best_map', 0.0)
            self.global_step = checkpoint.get('global_step', 0)
            
            # Load training history
            if 'training_history' in checkpoint:
                self.training_history = defaultdict(list, checkpoint['training_history'])
            
            logger.info(f"Successfully loaded checkpoint from epoch {self.current_epoch}")
            logger.info(f"Resumed with best mAP: {self.best_map:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            raise
    
    def train(self):
        """
        Main training loop
        """
        logger.info("Starting training...")
        logger.info(f"Training for {self.num_epochs} epochs")
        logger.info(f"Starting from epoch {self.current_epoch}")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Update training history
            for key, value in train_metrics.items():
                self.training_history[key].append(value)
            
            # Validate epoch
            val_metrics = {}
            if epoch % self.validate_every == 0:
                val_metrics = self.validate_epoch()
                
                # Update training history
                for key, value in val_metrics.items():
                    self.training_history[key].append(value)
            
            # Save training history after each epoch (incremental save)
            self._save_training_history()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # Use validation mAP for ReduceLROnPlateau
                    if 'val/coco_mAP' in val_metrics:
                        self.scheduler.step(val_metrics['val/coco_mAP'])
                    else:
                        logger.warning("No validation mAP available for scheduler")
                else:
                    self.scheduler.step()
            
            # Check if best model
            current_map = val_metrics.get('val/coco_mAP', 0.0)
            is_best = current_map > self.best_map
            if is_best:
                self.best_map = current_map
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Log metrics
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics['epoch'] = epoch
            all_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            all_metrics['epoch_time'] = epoch_time
            
            # Enhanced logging
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics.get('train/total_loss', 0):.4f} | "
                f"Val mAP: {current_map:.4f} | "
                f"Best mAP: {self.best_map:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Log to wandb
            if wandb is not None and wandb.run:
                wandb.log(all_metrics)
            
            # Save checkpoint
            if epoch % self.save_every == 0 or is_best or epoch == self.num_epochs - 1:
                self.save_checkpoint(epoch, is_best)
        
        # Final save of training history with completion status
        history_path = self.log_dir / 'training_history.json'
        try:
            # Load existing history and mark as completed
            with open(history_path, 'r') as f:
                history_dict = json.load(f)
            
            # Update metadata to mark training as completed
            if '_metadata' not in history_dict:
                history_dict['_metadata'] = {}
            
            history_dict['_metadata'].update({
                'training_completed': True,
                'completion_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_epochs': self.num_epochs,
                'final_best_map': self.best_map
            })
            
            # Save final version
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to update final training history: {e}")
            # Fallback to basic save
            self._save_training_history()
        
        logger.info("Training completed!")
        logger.info(f"Best validation mAP achieved: {self.best_map:.4f}")
        logger.info(f"Training history saved to: {history_path}")
        logger.info(f"All training outputs saved in: {self.output_dir}")
        logger.info(f"  - Configuration: {self.output_dir / f'{self.config_name}_config.yaml'}")
        logger.info(f"  - Best model: {self.checkpoint_dir / 'best_model.pth'}")
        logger.info(f"  - Latest model: {self.checkpoint_dir / 'latest_model.pth'}")
        logger.info(f"  - Training history: {history_path}")
        
        # Final evaluation on test set if available
        if self.test_loader and self.evaluator:
            try:
                logger.info("Running final evaluation on test set...")
                # TODO: Implement test evaluation if needed
                pass
            except Exception as e:
                logger.warning(f"Test evaluation failed: {e}")


def train_model(config: Dict[str, Any]) -> ObjectDetectionTrainer:
    """
    Convenience function to train model with configuration
    
    Args:
        config: Complete configuration dictionary
        
    Returns:
        Trained trainer instance
    """
    trainer = ObjectDetectionTrainer(config)
    trainer.train()
    return trainer