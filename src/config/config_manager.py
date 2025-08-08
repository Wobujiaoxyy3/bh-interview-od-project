"""
Configuration Manager for Floor Plan Object Detection
Handles loading, merging, and validation of YAML configuration files
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Custom exception for configuration-related errors"""
    pass


class ConfigManager:
    """
    Configuration manager that handles loading and merging of YAML config files
    Supports hierarchical configuration with base config and model-specific overrides
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize configuration manager
        
        Args:
            project_root: Path to project root directory
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        
        self.project_root = Path(project_root)
        self.config_dir = self.project_root / "configs"
        self.base_config_path = self.config_dir / "base_config.yaml"
        
        if not self.config_dir.exists():
            raise ConfigError(f"Config directory not found: {self.config_dir}")
            
        if not self.base_config_path.exists():
            raise ConfigError(f"Base config file not found: {self.base_config_path}")
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load configuration by merging base config with model-specific config
        
        Args:
            config_name: Name of the model-specific config file (without .yaml extension)
            
        Returns:
            Merged configuration dictionary
        """
        # Load base configuration
        base_config = self._load_yaml_file(self.base_config_path)
        
        # Load model-specific configuration
        model_config_path = self.config_dir / f"{config_name}.yaml"
        if not model_config_path.exists():
            raise ConfigError(f"Model config file not found: {model_config_path}")
            
        model_config = self._load_yaml_file(model_config_path)
        
        # Merge configurations
        merged_config = self._deep_merge(base_config, model_config)
        
        # Validate configuration
        self._validate_config(merged_config)
        
        # Add metadata
        merged_config['_meta'] = {
            'config_name': config_name,
            'base_config_path': str(self.base_config_path),
            'model_config_path': str(model_config_path),
            'project_root': str(self.project_root)
        }
        
        logger.info(f"Loaded configuration: {config_name}")
        return merged_config
    
    def save_config(self, config: Dict[str, Any], save_path: Path) -> None:
        """
        Save configuration to YAML file
        
        Args:
            config: Configuration dictionary to save
            save_path: Path where to save the configuration
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove metadata before saving
        config_to_save = deepcopy(config)
        config_to_save.pop('_meta', None)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, indent=2, allow_unicode=True)
            
        logger.info(f"Saved configuration to: {save_path}")
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML file and return as dictionary"""
        # Try different encodings to handle Windows encoding issues
        for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return yaml.safe_load(f) or {}
            except UnicodeDecodeError:
                continue
            except yaml.YAMLError as e:
                raise ConfigError(f"Error parsing YAML file {file_path}: {e}")
            except FileNotFoundError:
                raise ConfigError(f"Config file not found: {file_path}")
        
        # If all encodings fail, raise error
        raise ConfigError(f"Could not decode file {file_path} with any supported encoding (tried: utf-8, utf-8-sig, gbk, cp1252)")
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with override taking precedence
        
        Args:
            base: Base dictionary
            override: Dictionary with override values
            
        Returns:
            Merged dictionary
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
                
        return result
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration structure and required fields
        
        Args:
            config: Configuration dictionary to validate
        """
        required_sections = ['project', 'data', 'model', 'training', 'evaluation']
        
        for section in required_sections:
            if section not in config:
                raise ConfigError(f"Required configuration section missing: {section}")
        
        # Validate data section
        data_config = config['data']
        if 'images_dir' not in data_config:
            raise ConfigError("Missing 'images_dir' in data configuration")
        if 'annotations_file' not in data_config:
            raise ConfigError("Missing 'annotations_file' in data configuration")
            
        # Validate model section
        model_config = config['model']
        if 'type' not in model_config and 'architecture' not in model_config:
            raise ConfigError("Missing 'type' or 'architecture' in model configuration")
            
        # Validate training section
        training_config = config['training']
        required_training_fields = ['num_epochs']
        for field in required_training_fields:
            if field not in training_config:
                raise ConfigError(f"Missing '{field}' in training configuration")
        
        # Validate optimizer section exists
        if 'optimizer' not in config:
            raise ConfigError("Missing 'optimizer' section in configuration")
            
        # Validate data_loader section exists (batch_size is here, not in training)
        if 'data_loader' not in config:
            raise ConfigError("Missing 'data_loader' section in configuration")
            
        data_loader_config = config['data_loader']
        if 'batch_size' not in data_loader_config:
            raise ConfigError("Missing 'batch_size' in data_loader configuration")
        
        # Validate paths exist
        project_root = Path(config.get('_meta', {}).get('project_root', self.project_root))
        
        images_dir = project_root / data_config['images_dir']
        if not images_dir.exists():
            logger.warning(f"Images directory does not exist: {images_dir}")
            
        annotations_file = project_root / data_config['annotations_file'] 
        if not annotations_file.exists():
            logger.warning(f"Annotations file does not exist: {annotations_file}")
    
    def get_available_configs(self) -> List[str]:
        """
        Get list of available configuration files
        
        Returns:
            List of configuration names (without .yaml extension)
        """
        config_files = []
        for file_path in self.config_dir.glob("*.yaml"):
            if file_path.name != "base_config.yaml":
                config_files.append(file_path.stem)
        return sorted(config_files)
    
    def update_config_value(self, config: Dict[str, Any], key_path: str, value: Any) -> Dict[str, Any]:
        """
        Update a configuration value using dot notation path
        
        Args:
            config: Configuration dictionary
            key_path: Dot-separated path to the configuration key (e.g., "training.optimizer.lr")
            value: New value to set
            
        Returns:
            Updated configuration dictionary
        """
        config = deepcopy(config)
        keys = key_path.split('.')
        
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
            
        current[keys[-1]] = value
        return config


def load_config(config_name: str, project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Convenience function to load configuration
    
    Args:
        config_name: Name of the configuration to load
        project_root: Path to project root directory
        
    Returns:
        Configuration dictionary
    """
    manager = ConfigManager(project_root)
    return manager.load_config(config_name)


def get_config_manager(project_root: Optional[Path] = None) -> ConfigManager:
    """
    Get configuration manager instance
    
    Args:
        project_root: Path to project root directory
        
    Returns:
        ConfigManager instance
    """
    return ConfigManager(project_root)