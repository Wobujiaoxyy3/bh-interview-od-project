"""
Random seed utilities for reproducible experiments
"""

import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducible results
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        # Make CuDNN deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set random seed to {seed}")


def get_random_state():
    """
    Get current random state for all generators
    
    Returns:
        Dictionary with random states
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state()
    }
    
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state()
    
    return state


def set_random_state(state):
    """
    Restore random state for all generators
    
    Args:
        state: Dictionary with random states from get_random_state()
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if torch.cuda.is_available() and 'torch_cuda' in state:
        torch.cuda.set_rng_state(state['torch_cuda'])