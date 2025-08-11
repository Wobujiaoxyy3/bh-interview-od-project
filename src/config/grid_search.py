"""
Grid Search Manager for Hyperparameter Optimization
Handles automated hyperparameter search across different parameter combinations
"""

import itertools
import logging
from typing import Dict, List, Any, Iterator, Tuple
from pathlib import Path
import json
from copy import deepcopy

from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


class GridSearchManager:
    """
    Manager for grid search hyperparameter optimization
    Generates parameter combinations and manages experiment configurations
    """
    
    def __init__(self, base_config: Dict[str, Any], config_manager: ConfigManager):
        """
        Initialize grid search manager
        
        Args:
            base_config: Base configuration dictionary
            config_manager: ConfigManager instance for handling configurations
        """
        self.base_config = deepcopy(base_config)
        self.config_manager = config_manager
        self.search_params = base_config.get('grid_search', {}).get('parameters', [])
        
        if not self.search_params:
            logger.warning("No grid search parameters defined in configuration")
    
    def generate_configs(self) -> Iterator[Tuple[Dict[str, Any], str]]:
        """
        Generate all possible configuration combinations for grid search
        
        Yields:
            Tuple of (configuration_dict, experiment_name)
        """
        if not self.search_params:
            # No grid search parameters, return base config
            yield self.base_config, "base"
            return
        
        # Extract parameter names and values
        param_names = [param['name'] for param in self.search_params]
        param_values = [param['values'] for param in self.search_params]
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        logger.info(f"Generating {len(combinations)} configuration combinations")
        
        for i, combination in enumerate(combinations):
            # Create new configuration
            config = deepcopy(self.base_config)
            
            # Update parameters
            experiment_params = {}
            for param_name, param_value in zip(param_names, combination):
                config = self.config_manager.update_config_value(config, param_name, param_value)
                experiment_params[param_name] = param_value
            
            # Generate experiment name
            experiment_name = self._generate_experiment_name(experiment_params, i)
            
            # Add experiment metadata
            config['_meta']['experiment'] = {
                'name': experiment_name,
                'parameters': experiment_params,
                'combination_id': i
            }
            
            yield config, experiment_name
    
    def get_total_combinations(self) -> int:
        """
        Get total number of parameter combinations
        
        Returns:
            Number of combinations
        """
        if not self.search_params:
            return 1
            
        total = 1
        for param in self.search_params:
            total *= len(param['values'])
        return total
    
    def _generate_experiment_name(self, params: Dict[str, Any], combination_id: int) -> str:
        """
        Generate a readable experiment name from parameters
        
        Args:
            params: Dictionary of parameter name-value pairs
            combination_id: Unique ID for this combination
            
        Returns:
            Generated experiment name
        """
        # Create short parameter representations
        param_parts = []
        for param_name, param_value in params.items():
            # Extract the last part of the parameter path for brevity
            short_name = param_name.split('.')[-1]
            
            # Format value for readability
            if isinstance(param_value, float):
                value_str = f"{param_value:.4f}".rstrip('0').rstrip('.')
            else:
                value_str = str(param_value)
            
            param_parts.append(f"{short_name}_{value_str}")
        
        if param_parts:
            return f"exp_{combination_id:03d}_{'_'.join(param_parts)}"
        else:
            return f"exp_{combination_id:03d}"
    
    def save_search_results(self, results: List[Dict[str, Any]], save_path: Path) -> None:
        """
        Save grid search results to file
        
        Args:
            results: List of experiment results
            save_path: Path to save results
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for saving
        search_summary = {
            'total_experiments': len(results),
            'search_parameters': self.search_params,
            'results': results
        }
        
        with open(save_path, 'w') as f:
            json.dump(search_summary, f, indent=2, default=str)
        
        logger.info(f"Grid search results saved to: {save_path}")
    
    def get_best_config(self, results: List[Dict[str, Any]], 
                       metric: str = "mAP@0.5") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get the best configuration based on a metric
        
        Args:
            results: List of experiment results
            metric: Metric to optimize for
            
        Returns:
            Tuple of (best_config, best_result)
        """
        if not results:
            raise ValueError("No results provided")
        
        # Find best result
        best_result = max(results, key=lambda x: x.get('metrics', {}).get(metric, 0))
        
        # Reconstruct best configuration
        best_params = best_result.get('experiment', {}).get('parameters', {})
        best_config = deepcopy(self.base_config)
        
        for param_name, param_value in best_params.items():
            best_config = self.config_manager.update_config_value(best_config, param_name, param_value)
        
        return best_config, best_result


class GridSearchRunner:
    """
    Runner for executing grid search experiments
    """
    
    def __init__(self, config_name: str, project_root: Path = None):
        """
        Initialize grid search runner
        
        Args:
            config_name: Name of base configuration
            project_root: Path to project root
        """
        self.config_manager = ConfigManager(project_root)
        self.base_config = self.config_manager.load_config(config_name)
        self.grid_search = GridSearchManager(self.base_config, self.config_manager)
        
        # Check if grid search is enabled
        if not self.base_config.get('grid_search', {}).get('enabled', False):
            logger.warning("Grid search is not enabled in configuration")
    
    def run_search(self, train_function, results_dir: Path = None) -> List[Dict[str, Any]]:
        """
        Run grid search experiments
        
        Args:
            train_function: Function to call for training (should accept config and return results)
            results_dir: Directory to save results
            
        Returns:
            List of experiment results
        """
        if results_dir is None:
            results_dir = Path("grid_search_results")
        
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        total_experiments = self.grid_search.get_total_combinations()
        
        logger.info(f"Starting grid search with {total_experiments} experiments")
        
        for i, (config, experiment_name) in enumerate(self.grid_search.generate_configs()):
            logger.info(f"Running experiment {i+1}/{total_experiments}: {experiment_name}")
            
            try:
                # Run training
                result = train_function(config)
                
                # Add experiment metadata to result
                result['experiment'] = config['_meta']['experiment']
                result['success'] = True
                
                results.append(result)
                
                # Save individual experiment config and result
                exp_dir = results_dir / experiment_name
                exp_dir.mkdir(exist_ok=True)
                
                self.config_manager.save_config(config, exp_dir / "config.yaml")
                
                with open(exp_dir / "results.json", 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                
                logger.info(f"Experiment {experiment_name} completed successfully")
                
            except Exception as e:
                logger.error(f"Experiment {experiment_name} failed: {str(e)}")
                
                # Record failure
                failed_result = {
                    'experiment': config['_meta']['experiment'],
                    'success': False,
                    'error': str(e),
                    'metrics': {}
                }
                results.append(failed_result)
        
        # Save overall results
        self.grid_search.save_search_results(results, results_dir / "grid_search_summary.json")
        
        # Log summary
        successful_experiments = sum(1 for r in results if r.get('success', False))
        logger.info(f"Grid search completed: {successful_experiments}/{total_experiments} experiments successful")
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]], 
                       metric: str = "mAP@0.5") -> Dict[str, Any]:
        """
        Analyze grid search results
        
        Args:
            results: List of experiment results
            metric: Metric to analyze
            
        Returns:
            Analysis summary
        """
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {'error': 'No successful experiments'}
        
        # Get metric values
        metric_values = [r.get('metrics', {}).get(metric, 0) for r in successful_results]
        
        # Find best and worst
        best_config, best_result = self.grid_search.get_best_config(successful_results, metric)
        worst_result = min(successful_results, key=lambda x: x.get('metrics', {}).get(metric, 0))
        
        analysis = {
            'total_experiments': len(results),
            'successful_experiments': len(successful_results),
            'metric': metric,
            'best_value': max(metric_values),
            'worst_value': min(metric_values),
            'mean_value': sum(metric_values) / len(metric_values),
            'best_experiment': best_result.get('experiment', {}),
            'worst_experiment': worst_result.get('experiment', {}),
            'parameter_analysis': self._analyze_parameters(successful_results, metric)
        }
        
        return analysis
    
    def _analyze_parameters(self, results: List[Dict[str, Any]], 
                           metric: str) -> Dict[str, Any]:
        """
        Analyze the impact of different parameters on the metric
        
        Args:
            results: Successful experiment results
            metric: Metric to analyze
            
        Returns:
            Parameter analysis
        """
        if not self.grid_search.search_params:
            return {}
        
        analysis = {}
        
        for param_info in self.grid_search.search_params:
            param_name = param_info['name']
            param_values = param_info['values']
            
            # Group results by parameter value
            value_groups = {value: [] for value in param_values}
            
            for result in results:
                param_value = result.get('experiment', {}).get('parameters', {}).get(param_name)
                if param_value in value_groups:
                    metric_value = result.get('metrics', {}).get(metric, 0)
                    value_groups[param_value].append(metric_value)
            
            # Calculate statistics for each parameter value
            param_analysis = {}
            for value, metrics in value_groups.items():
                if metrics:
                    param_analysis[str(value)] = {
                        'mean': sum(metrics) / len(metrics),
                        'max': max(metrics),
                        'min': min(metrics),
                        'count': len(metrics)
                    }
            
            analysis[param_name] = param_analysis
        
        return analysis