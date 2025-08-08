"""
Data Splitter for Floor Plan Object Detection Dataset
Implements stratified splitting to maintain class distribution across splits
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import random

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Handles stratified splitting of COCO format dataset
    Ensures balanced class distribution across train/validation/test splits
    """
    
    def __init__(self, annotations_file: Path, train_ratio: float = 0.7, 
                 val_ratio: float = 0.2, test_ratio: float = 0.1, random_seed: int = 42):
        """
        Initialize data splitter
        
        Args:
            annotations_file: Path to COCO format annotations file
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation  
            test_ratio: Proportion of data for testing
            random_seed: Random seed for reproducibility
        """
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
            
        self.annotations_file = Path(annotations_file)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Load data
        self.coco_data = self._load_annotations()
        self.image_annotations = self._group_annotations_by_image()
        
        logger.info(f"Loaded {len(self.coco_data['images'])} images with "
                   f"{len(self.coco_data['annotations'])} annotations")
    
    def _load_annotations(self) -> Dict[str, Any]:
        """Load COCO format annotations"""
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
            
        with open(self.annotations_file, 'r') as f:
            return json.load(f)
    
    def _group_annotations_by_image(self) -> Dict[int, List[Dict[str, Any]]]:
        """Group annotations by image ID"""
        image_annotations = defaultdict(list)
        for annotation in self.coco_data['annotations']:
            image_annotations[annotation['image_id']].append(annotation)
        return dict(image_annotations)
    
    def _get_image_class_distribution(self) -> Dict[int, List[int]]:
        """
        Get class distribution for each image
        
        Returns:
            Dictionary mapping image_id to list of present class IDs
        """
        image_classes = {}
        
        for image in self.coco_data['images']:
            image_id = image['id']
            annotations = self.image_annotations.get(image_id, [])
            
            # Get unique class IDs present in this image
            class_ids = list(set(ann['category_id'] for ann in annotations))
            image_classes[image_id] = sorted(class_ids)
            
        return image_classes
    
    def _create_stratification_key(self, class_ids: List[int]) -> str:
        """
        Create stratification key from class IDs
        For multi-label stratification, we use class combinations
        
        Args:
            class_ids: List of class IDs present in image
            
        Returns:
            String key representing class combination
        """
        return "_".join(map(str, sorted(class_ids)))
    
    def _stratified_split_images(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Perform stratified split of images based on class distribution
        
        Returns:
            Tuple of (train_image_ids, val_image_ids, test_image_ids)
        """
        image_classes = self._get_image_class_distribution()
        
        # Create stratification keys
        image_ids = list(image_classes.keys())
        stratification_keys = [self._create_stratification_key(image_classes[img_id]) 
                              for img_id in image_ids]
        
        # Count occurrences of each stratification key
        key_counts = Counter(stratification_keys)
        logger.info(f"Found {len(key_counts)} unique class combinations")
        
        # Log class combination statistics
        logger.info("Class combination distribution:")
        for key, count in sorted(key_counts.items()):
            logger.info(f"  Class combination '{key}': {count} images")
            
        # Check for stratification viability
        single_sample_keys = [key for key, count in key_counts.items() if count == 1]
        if single_sample_keys:
            logger.warning(f"Found {len(single_sample_keys)} class combinations with only 1 sample: {single_sample_keys}")
            logger.info("These will be handled as rare samples and distributed randomly.")
        
        # Handle rare combinations (with only 1 or 2 images)
        # For stratified split to work, we need at least 2 samples per group
        rare_threshold = 2
        rare_images = []
        common_image_ids = []
        common_keys = []
        
        for img_id, key in zip(image_ids, stratification_keys):
            if key_counts[key] < rare_threshold:
                rare_images.append(img_id)
            else:
                common_image_ids.append(img_id)
                common_keys.append(key)
        
        logger.info(f"Stratification strategy:")
        logger.info(f"  Common samples (>={rare_threshold} per group): {len(common_image_ids)} images - will use stratified split")
        logger.info(f"  Rare samples (<{rare_threshold} per group): {len(rare_images)} images - will distribute randomly")
        
        # Split common images using stratification
        if common_image_ids:
            try:
                # First split: train vs (val + test)
                train_ids, val_test_ids, train_keys, val_test_keys = train_test_split(
                    common_image_ids, common_keys,
                    test_size=(self.val_ratio + self.test_ratio),
                    stratify=common_keys,
                    random_state=self.random_seed
                )
                
                # Second split: val vs test
                if len(val_test_ids) > 1:
                    try:
                        relative_test_size = self.test_ratio / (self.val_ratio + self.test_ratio)
                        
                        val_ids, test_ids = train_test_split(
                            val_test_ids,
                            test_size=relative_test_size,
                            stratify=val_test_keys,
                            random_state=self.random_seed
                        )
                    except ValueError as e:
                        logger.warning(f"Cannot stratify val/test split: {e}. Using random split.")
                        relative_test_size = self.test_ratio / (self.val_ratio + self.test_ratio)
                        val_ids, test_ids = train_test_split(
                            val_test_ids,
                            test_size=relative_test_size,
                            random_state=self.random_seed
                        )
                else:
                    val_ids = val_test_ids
                    test_ids = []
                    
            except ValueError as e:
                logger.warning(f"Cannot perform stratified split: {e}. Using random split for common images.")
                # Fallback to random split
                train_ids, val_test_ids = train_test_split(
                    common_image_ids,
                    test_size=(self.val_ratio + self.test_ratio),
                    random_state=self.random_seed
                )
                
                if len(val_test_ids) > 1:
                    relative_test_size = self.test_ratio / (self.val_ratio + self.test_ratio)
                    val_ids, test_ids = train_test_split(
                        val_test_ids,
                        test_size=relative_test_size,
                        random_state=self.random_seed
                    )
                else:
                    val_ids = val_test_ids
                    test_ids = []
        else:
            train_ids, val_ids, test_ids = [], [], []
        
        # Distribute rare images randomly
        if rare_images:
            random.shuffle(rare_images)
            
            n_train_rare = int(len(rare_images) * self.train_ratio)
            n_val_rare = int(len(rare_images) * self.val_ratio)
            
            train_ids.extend(rare_images[:n_train_rare])
            val_ids.extend(rare_images[n_train_rare:n_train_rare + n_val_rare])
            test_ids.extend(rare_images[n_train_rare + n_val_rare:])
            
            logger.info(f"Distributed {len(rare_images)} rare images randomly")
        
        return train_ids, val_ids, test_ids
    
    def _create_split_annotations(self, image_ids: List[int], split_name: str) -> Dict[str, Any]:
        """
        Create COCO format annotations for a specific split
        
        Args:
            image_ids: List of image IDs for this split
            split_name: Name of the split (train/val/test)
            
        Returns:
            COCO format dictionary for this split
        """
        image_id_set = set(image_ids)
        
        # Filter images
        split_images = [img for img in self.coco_data['images'] if img['id'] in image_id_set]
        
        # Filter annotations
        split_annotations = [ann for ann in self.coco_data['annotations'] 
                           if ann['image_id'] in image_id_set]
        
        # Create new annotation IDs (starting from 1)
        for i, annotation in enumerate(split_annotations):
            annotation['id'] = i + 1
        
        # Create split dictionary
        split_data = {
            'info': self.coco_data['info'].copy(),
            'categories': self.coco_data['categories'].copy(),
            'images': split_images,
            'annotations': split_annotations
        }
        
        # Update info
        split_data['info']['description'] += f" - {split_name.upper()} split"
        
        return split_data
    
    def _analyze_split_distribution(self, train_ids: List[int], val_ids: List[int], 
                                  test_ids: List[int]) -> Dict[str, Any]:
        """
        Analyze class distribution across splits
        
        Returns:
            Analysis dictionary with distribution statistics
        """
        def get_class_counts(image_ids: List[int]) -> Dict[int, int]:
            class_counts = defaultdict(int)
            for image_id in image_ids:
                annotations = self.image_annotations.get(image_id, [])
                for annotation in annotations:
                    class_counts[annotation['category_id']] += 1
            return dict(class_counts)
        
        train_classes = get_class_counts(train_ids)
        val_classes = get_class_counts(val_ids)
        test_classes = get_class_counts(test_ids)
        
        # Calculate total counts
        total_classes = defaultdict(int)
        for counts in [train_classes, val_classes, test_classes]:
            for class_id, count in counts.items():
                total_classes[class_id] += count
        
        # Calculate proportions
        analysis = {
            'total_images': len(train_ids) + len(val_ids) + len(test_ids),
            'train_images': len(train_ids),
            'val_images': len(val_ids),
            'test_images': len(test_ids),
            'class_distribution': {}
        }
        
        for class_id in total_classes.keys():
            train_count = train_classes.get(class_id, 0)
            val_count = val_classes.get(class_id, 0)
            test_count = test_classes.get(class_id, 0)
            total_count = total_classes[class_id]
            
            analysis['class_distribution'][class_id] = {
                'total': total_count,
                'train': {'count': train_count, 'ratio': train_count / total_count if total_count > 0 else 0},
                'val': {'count': val_count, 'ratio': val_count / total_count if total_count > 0 else 0},
                'test': {'count': test_count, 'ratio': test_count / total_count if total_count > 0 else 0}
            }
        
        return analysis
    
    def split_dataset(self, output_dir: Path) -> Dict[str, Any]:
        """
        Split dataset and save splits to separate files
        
        Args:
            output_dir: Directory to save split files
            
        Returns:
            Analysis of the split distribution
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting stratified dataset split...")
        
        # Perform stratified split
        train_ids, val_ids, test_ids = self._stratified_split_images()
        
        logger.info(f"Split sizes - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
        
        # Create split annotations
        splits = {
            'train': self._create_split_annotations(train_ids, 'train'),
            'val': self._create_split_annotations(val_ids, 'val'),
            'test': self._create_split_annotations(test_ids, 'test')
        }
        
        # Save splits
        split_files = {}
        for split_name, split_data in splits.items():
            output_file = output_dir / f"{split_name}_annotations.json"
            with open(output_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            split_files[split_name] = str(output_file)
            logger.info(f"Saved {split_name} split to {output_file}")
        
        # Analyze distribution
        analysis = self._analyze_split_distribution(train_ids, val_ids, test_ids)
        analysis['split_files'] = split_files
        
        # Save analysis
        analysis_file = output_dir / "split_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Log distribution
        self._log_distribution_analysis(analysis)
        
        logger.info(f"Dataset split completed. Files saved to {output_dir}")
        
        return analysis
    
    def _log_distribution_analysis(self, analysis: Dict[str, Any]) -> None:
        """Log distribution analysis"""
        logger.info("=== Split Distribution Analysis ===")
        logger.info(f"Total images: {analysis['total_images']}")
        logger.info(f"Train: {analysis['train_images']} ({analysis['train_images']/analysis['total_images']:.1%})")
        logger.info(f"Val: {analysis['val_images']} ({analysis['val_images']/analysis['total_images']:.1%})")
        logger.info(f"Test: {analysis['test_images']} ({analysis['test_images']/analysis['total_images']:.1%})")
        
        logger.info("\nClass distribution across splits:")
        for class_id, dist in analysis['class_distribution'].items():
            logger.info(f"Class {class_id}:")
            logger.info(f"  Train: {dist['train']['count']} ({dist['train']['ratio']:.1%})")
            logger.info(f"  Val: {dist['val']['count']} ({dist['val']['ratio']:.1%})")
            logger.info(f"  Test: {dist['test']['count']} ({dist['test']['ratio']:.1%})")


def split_dataset(annotations_file: Path, output_dir: Path, 
                 train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1,
                 random_seed: int = 42) -> Dict[str, Any]:
    """
    Convenience function to split dataset
    
    Args:
        annotations_file: Path to COCO annotations file
        output_dir: Output directory for split files
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility
        
    Returns:
        Split analysis dictionary
    """
    splitter = DataSplitter(annotations_file, train_ratio, val_ratio, test_ratio, random_seed)
    return splitter.split_dataset(output_dir)