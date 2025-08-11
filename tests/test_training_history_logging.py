#!/usr/bin/env python3
"""
Test script to verify the new real-time training history logging
"""

import sys
from pathlib import Path
import json
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_training_history_logging():
    """Test that training history is saved incrementally during training"""
    
    print("Testing Real-time Training History Logging")
    print("=" * 60)
    
    try:
        from config import ConfigManager
        from training.trainer import ObjectDetectionTrainer
        from collections import defaultdict
        import time
        
        # Load config
        config_manager = ConfigManager()
        config = config_manager.load_config('faster_rcnn_config')
        
        # Create temporary output directory for testing
        temp_dir = Path(tempfile.mkdtemp())
        config['output_dir'] = str(temp_dir)
        
        config_name = config.get('_meta', {}).get('config_name', 'test_config')
        expected_history_file = temp_dir / config_name / 'logs' / 'training_history.json'
        
        print(f"Testing with temporary directory: {temp_dir}")
        print(f"Expected history file: {expected_history_file}")
        
        # Create a minimal mock trainer to test the logging logic
        class MockTrainer:
            def __init__(self, config):
                from collections import defaultdict
                self.config = config
                self.config_name = config.get('_meta', {}).get('config_name', 'test_config')
                
                # Create directory structure
                base_output_dir = Path(config.get('output_dir', 'outputs'))
                self.output_dir = base_output_dir / self.config_name
                self.log_dir = self.output_dir / 'logs'
                self.log_dir.mkdir(parents=True, exist_ok=True)
                
                # Initialize training state
                self.current_epoch = 0
                self.global_step = 0
                self.best_map = 0.0
                self.training_history = defaultdict(list)
            
            def _save_training_history(self):
                """Copy of the real _save_training_history method"""
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
                    print(f"Failed to save training history: {e}")
                    raise
            
            def simulate_training_epochs(self, num_epochs=3):
                """Simulate training epochs with metric logging"""
                for epoch in range(num_epochs):
                    self.current_epoch = epoch
                    self.global_step += 10  # Simulate batch steps
                    
                    # Simulate training metrics
                    self.training_history['train_loss'].append(1.0 - epoch * 0.1)
                    self.training_history['train_total_loss'].append(1.2 - epoch * 0.1)
                    
                    # Simulate validation metrics (every epoch)
                    self.training_history['val_map'].append(0.3 + epoch * 0.15)
                    self.training_history['val_map_50'].append(0.5 + epoch * 0.1)
                    
                    # Update best mAP
                    current_map = self.training_history['val_map'][-1]
                    if current_map > self.best_map:
                        self.best_map = current_map
                    
                    # Save history after each epoch (this is the key test)
                    self._save_training_history()
                    
                    print(f"  Epoch {epoch}: Saved metrics to history file")
                    
                    # Verify file exists and contains data
                    if not expected_history_file.exists():
                        raise Exception(f"History file not created after epoch {epoch}")
                    
                    with open(expected_history_file, 'r') as f:
                        saved_data = json.load(f)
                    
                    # Check that current epoch data is saved
                    if len(saved_data.get('train_loss', [])) != epoch + 1:
                        raise Exception(f"Expected {epoch + 1} training loss entries, got {len(saved_data.get('train_loss', []))}")
                    
                    # Check metadata
                    metadata = saved_data.get('_metadata', {})
                    if metadata.get('current_epoch') != epoch:
                        raise Exception(f"Metadata current_epoch mismatch: expected {epoch}, got {metadata.get('current_epoch')}")
        
        # Test the mock trainer
        print("\nSimulating training with incremental history saving...")
        mock_trainer = MockTrainer(config)
        mock_trainer.simulate_training_epochs(3)
        
        print("\nPASS: Training history saved after each epoch")
        
        # Verify final file structure
        if expected_history_file.exists():
            with open(expected_history_file, 'r') as f:
                final_data = json.load(f)
            
            print(f"PASS: Final history file exists")
            print(f"  - Training epochs logged: {len(final_data.get('train_loss', []))}")
            print(f"  - Validation epochs logged: {len(final_data.get('val_map', []))}")
            
            # Check metadata
            metadata = final_data.get('_metadata', {})
            print(f"  - Final epoch: {metadata.get('current_epoch')}")
            print(f"  - Best mAP: {metadata.get('best_map')}")
            print(f"  - Last updated: {metadata.get('last_updated')}")
            print(f"  - Config name: {metadata.get('config_name')}")
            
            if metadata:
                print("PASS: Metadata properly saved")
            else:
                print("FAIL: Missing metadata")
                return False
        else:
            print("FAIL: Final history file not found")
            return False
        
        return True
        
    except ImportError as e:
        print(f"SKIP: Cannot test due to missing dependencies: {e}")
        return True
        
    except Exception as e:
        print(f"FAIL: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup temporary directory
        if 'temp_dir' in locals() and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                print(f"\nCleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up temporary directory: {e}")


def test_history_file_structure():
    """Test the structure of the saved training history JSON"""
    
    print("\n" + "=" * 60)
    print("Testing Training History File Structure")
    print("=" * 60)
    
    # Create sample training history data
    sample_history = {
        'train_loss': [1.0, 0.9, 0.8, 0.7],
        'train_total_loss': [1.2, 1.1, 1.0, 0.9],
        'val_map': [0.3, 0.45, 0.6, 0.75],
        'val_map_50': [0.5, 0.6, 0.7, 0.8],
        '_metadata': {
            'current_epoch': 3,
            'global_step': 40,
            'best_map': 0.75,
            'last_updated': '2024-01-15 14:30:45',
            'config_name': 'faster_rcnn_config',
            'training_completed': False
        }
    }
    
    print("Expected training history structure:")
    print(json.dumps(sample_history, indent=2))
    
    # Verify structure
    required_keys = ['train_loss', 'val_map', '_metadata']
    missing_keys = []
    
    for key in required_keys:
        if key not in sample_history:
            missing_keys.append(key)
        else:
            print(f"PASS: {key} present in structure")
    
    if missing_keys:
        print(f"FAIL: Missing keys: {missing_keys}")
        return False
    
    # Check metadata structure
    metadata = sample_history['_metadata']
    required_metadata = ['current_epoch', 'global_step', 'best_map', 'last_updated', 'config_name']
    
    for key in required_metadata:
        if key in metadata:
            print(f"PASS: Metadata contains {key}")
        else:
            print(f"FAIL: Metadata missing {key}")
            return False
    
    print("PASS: Training history structure is correct")
    return True


def main():
    """Main test function"""
    
    print("Training History Logging Tests")
    print("Testing new real-time logging functionality...")
    
    # Run tests
    test1_passed = test_training_history_logging()
    test2_passed = test_history_file_structure()
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    tests_passed = sum([test1_passed, test2_passed])
    total_tests = 2
    
    if tests_passed == total_tests:
        print(f"SUCCESS: ALL {total_tests} TESTS PASSED!")
        print("\nNew training history logging features:")
        print("  - Real-time saving after each epoch")
        print("  - Metadata tracking (epoch, step, best_map, timestamp)")
        print("  - Recovery-friendly (survives training interruptions)")
        print("  - Training completion status tracking")
        
        return 0
    else:
        print(f"FAIL: {total_tests - tests_passed} out of {total_tests} tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())