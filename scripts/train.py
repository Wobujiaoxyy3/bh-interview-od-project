"""
Main training script for floor plan object detection
Supports both Faster R-CNN and RetinaNet with grid search capabilities
"""

import sys
import argparse
import logging
from pathlib import Path
import yaml
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ConfigManager, GridSearchRunner
from training import train_model
from datasets import split_dataset
from utils import setup_logging, set_random_seed


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train floor plan object detection models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Grid search
    parser.add_argument('--grid-search', action='store_true',
                       help='Run grid search over hyperparameters')
    parser.add_argument('--grid-search-config', type=str, default=None,
                       help='Path to grid search configuration file')
    
    # Data splitting
    parser.add_argument('--split-data', action='store_true',
                       help='Split data before training')
    parser.add_argument('--force-split', action='store_true',
                       help='Force data splitting even if splits exist')
    
    # Output and logging
    parser.add_argument('--output-dir', '-o', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # Hardware
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of data loader workers')
    
    # Experiment tracking
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='floor-plan-detection',
                       help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='W&B run name')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Initial learning rate')
    
    return parser.parse_args()


def setup_device(args, config):
    """Setup computing device"""
    if args.device:
        device = args.device
    else:
        # Handle various device config formats
        device_config = config.get('device', 'auto')
        if isinstance(device_config, dict):
            device = device_config.get('type', 'auto')
        elif isinstance(device_config, str):
            device = device_config
        else:
            device = 'auto'
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Validate device
    if device.startswith('cuda') and not torch.cuda.is_available():
        logger = logging.getLogger(__name__)
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    config['device'] = device
    
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")
    
    if device == 'cuda' or device.startswith('cuda'):
        logger.info(f"CUDA devices available: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("Using CPU for training")


def override_config_from_args(config, args):
    """Override configuration with command line arguments"""
    
    # Training parameters
    if args.epochs is not None:
        config.setdefault('training', {})['num_epochs'] = args.epochs
    
    if args.batch_size is not None:
        config.setdefault('data_loader', {})['batch_size'] = args.batch_size
    
    if args.learning_rate is not None:
        config.setdefault('optimizer', {})['learning_rate'] = args.learning_rate
    
    if args.num_workers is not None:
        config.setdefault('data_loader', {})['num_workers'] = args.num_workers
    
    # Output directory
    config['output_dir'] = args.output_dir
    
    # Resume checkpoint
    if args.resume:
        config.setdefault('model', {})['checkpoint_path'] = args.resume
    
    # W&B configuration
    if args.wandb:
        config['use_wandb'] = True
        wandb_config = config.setdefault('wandb', {})
        wandb_config['project'] = args.wandb_project
        if args.wandb_run_name:
            wandb_config['run_name'] = args.wandb_run_name


def run_data_splitting(config, force_split=False):
    """Run data splitting if needed"""
    logger = logging.getLogger(__name__)
    
    data_config = config.get('data', {})
    splits_dir = Path(data_config.get('splits_dir', 'data/processed/splits'))
    annotations_file = Path(data_config.get('annotations_file', 
                                           'data/processed/2.0-coco_annotations_cleaned.json'))
    
    # Check if splits already exist
    train_file = splits_dir / 'train_annotations.json'
    val_file = splits_dir / 'val_annotations.json'
    test_file = splits_dir / 'test_annotations.json'
    
    if not force_split and all(f.exists() for f in [train_file, val_file, test_file]):
        logger.info("Data splits already exist. Skipping data splitting.")
        return
    
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    
    logger.info("Starting data splitting...")
    
    # Get split ratios from config
    split_config = config.get('data_splitting', {})
    train_ratio = split_config.get('train_ratio', 0.7)
    val_ratio = split_config.get('val_ratio', 0.2)
    test_ratio = split_config.get('test_ratio', 0.1)
    random_seed = config.get('random_seed', 42)
    
    # Run splitting
    analysis = split_dataset(
        annotations_file=annotations_file,
        output_dir=splits_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    
    logger.info("Data splitting completed!")
    logger.info(f"Train: {analysis['train_images']} images")
    logger.info(f"Val: {analysis['val_images']} images") 
    logger.info(f"Test: {analysis['test_images']} images")


def run_single_training(config):
    """Run single training experiment"""
    logger = logging.getLogger(__name__)
    logger.info("Starting single training run...")
    
    # Set random seed
    set_random_seed(config.get('random_seed', 42))
    
    # Run training
    trainer = train_model(config)
    
    logger.info("Training completed!")
    return trainer


def run_grid_search(config, grid_search_config_path=None):
    """Run grid search experiments"""
    logger = logging.getLogger(__name__)
    logger.info("Starting grid search...")
    
    # Load grid search configuration
    if grid_search_config_path:
        with open(grid_search_config_path, 'r') as f:
            grid_search_config = yaml.safe_load(f)
    else:
        grid_search_config = config.get('grid_search', {})
    
    if not grid_search_config:
        raise ValueError("Grid search configuration not found")
    
    # Create grid search runner
    runner = GridSearchRunner(config, grid_search_config)
    
    # Run grid search
    results = runner.run()
    
    logger.info("Grid search completed!")
    logger.info(f"Best configuration achieved mAP: {results['best_score']:.4f}")
    logger.info(f"Best configuration saved to: {results['best_config_path']}")
    
    return results


def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting floor plan object detection training")
    logger.info(f"Configuration file: {args.config}")
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        # Override config with command line arguments
        override_config_from_args(config, args)
        
        # Setup device
        setup_device(args, config)
        
        # Create output directory
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Save final configuration
        final_config_path = output_dir / 'config.yaml'
        with open(final_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        logger.info(f"Final configuration saved to: {final_config_path}")
        
        # Run data splitting if requested
        if args.split_data or not Path(config.get('data', {}).get('splits_dir', 'data/processed/splits')).exists():
            run_data_splitting(config, force_split=args.force_split)
        
        # Run training
        if args.grid_search:
            results = run_grid_search(config, args.grid_search_config)
        else:
            trainer = run_single_training(config)
        
        logger.info("All tasks completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()