#!/usr/bin/env python3
"""
Test script to verify basic framework structure (no external dependencies)
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    print("Testing basic framework structure...")
    
    print("\n1. Testing config manager...")
    from config.config_manager import ConfigManager
    print("+ ConfigManager imported successfully")
    
    print("\n2. Testing grid search...")
    from config.grid_search import GridSearchRunner
    print("+ GridSearchRunner imported successfully")
    
    print("\n3. Testing utils...")
    from utils.logging_utils import setup_logging
    from utils.seed_utils import set_random_seed
    print("+ Utils imported successfully")
    
    print("\n4. Testing basic config functionality...")
    # Test basic config manager functionality
    config_manager = ConfigManager()
    print("+ ConfigManager instantiated successfully")
    
    print("\n*** Basic framework structure is working! ***")
    print("\nTo use the full framework, install required packages:")
    print("uv add torch torchvision albumentations opencv-python pycocotools pyyaml")
    
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)