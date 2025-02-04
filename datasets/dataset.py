import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import openslide
import h5py


class BaseBagDataset(Dataset):
    def __init__(self, csv_path, slide_data=None):
        """
        Base class for Bag Datasets.

        Args:
            csv_path (str): Path to the CSV file containing slide information.
            slide_data (pd.DataFrame, optional): Preloaded DataFrame with slide information.
        """
        self.csv_path = csv_path
        self.slide_data = pd.read_csv(csv_path) if slide_data is None else slide_data

    def __len__(self):
        """Returns the number of slides in the dataset."""
        return len(self.slide_data)

    def get_split_from_df(self, all_splits, split_key='train', **kwargs):
        """
        Creates a dataset split based on the given key (train/val/test).

        Args:
            all_splits (pd.DataFrame): DataFrame containing split information.
            split_key (str, optional): Key for the split to extract. Defaults to 'train'.
            kwargs: Additional arguments for the dataset subclass.

        Returns:
            Subclass instance: A dataset object for the specified split.
        """
        split = all_splits[split_key].dropna().reset_index(drop=True)
        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())

            params = self.get_init_params()
            
            params.update(kwargs)

            return self.__class__(
                csv_path=self.csv_path,
                slide_data=self.slide_data[mask].reset_index(drop=True),
                **params
            )
        return None


    def get_init_params(self):
        """
        Returns a dictionary of parameters needed for initialization.
        Should be overridden by child classes.
        """
        return {}
    

    def return_splits(self, split_path, **kwargs):
        """
        Creates train, validation, and test splits from a CSV file.

        Args:
            split_path (str): Path to the CSV file containing split information.
            kwargs: Additional arguments for the dataset subclass.

        Returns:
            tuple: (train_split, val_split, test_split)
        """
        all_splits = pd.read_csv(split_path, dtype=self.slide_data['slide_id'].dtype)
        return (
            self.get_split_from_df(all_splits, 'train', **kwargs),
            self.get_split_from_df(all_splits, 'val', **kwargs),
            self.get_split_from_df(all_splits, 'test', **kwargs),
        )


class FeatureBagDataset(BaseBagDataset):
    def __init__(self, csv_path, features_dirs, bag_size=-1, is_test=False, is_train=False, slide_data=None):
        """
        Dataset for loading precomputed features.

        Args:
            features_dirs (list or str): Directory containing precomputed features.
            bag_size (int): Size of the bags for training/testing. -1 means load all.
            is_test (bool): Flag to indicate testing mode.
        """
        super().__init__(csv_path, slide_data)
        self.features_dirs = [features_dirs] if isinstance(features_dirs, str) else features_dirs
        self.bag_size = bag_size
        self.is_test = is_test
        self.is_train = is_train


    def get_init_params(self):
        """Returns parameters needed for initialization."""
        return {
            'features_dirs': self.features_dirs,
            'bag_size': self.bag_size,
            'is_test': self.is_test,
            'is_train': self.is_train
        }


    def get_feature_path(self, slide_id):
        """Finds the path for the feature file of a given slide ID."""
        if len(self.features_dirs) == 1:
            return os.path.join(self.features_dirs[0], 'pt_files', f'{slide_id}.pt')
        else:
            for feature_dir in self.features_dirs:
                path = os.path.join(feature_dir, 'pt_files', f'{slide_id}.pt')
                if os.path.exists(path):
                    return path

    def sample_features(self, features):
        """Samples features from the bag based on bag_size."""
        if self.bag_size == -1 or self.is_test:
            return features
        sampled_indices = np.random.choice(len(features), size=min(self.bag_size, len(features)), replace=False)
        if len(sampled_indices) < self.bag_size:
            padding = torch.zeros(self.bag_size - len(sampled_indices), features.shape[1])
            return torch.cat((features[sampled_indices], padding))
        return features[sampled_indices]


    def __getitem__(self, idx):
        """Returns a sampled bag of features, the label, and the slide ID."""
        slide_id = self.slide_data['slide_id'].iloc[idx]
        slide_label = self.slide_data['label'].iloc[idx]
        feature_dir = self.get_feature_path(slide_id)
        features = torch.load(feature_dir)
        return self.sample_features(features), torch.tensor(slide_label, dtype=torch.float32), slide_id


class PatchBagDataset(BaseBagDataset):
    def __init__(self, csv_path, data_dir, slide_dir, extension,
                 with_data_augmentation=False, max_bag_size=-1, is_train=False, is_test=False, slide_data=None, chunk_size=500):
        """
        Dataset for loading pre-extracted patches.

        Args:
            csv_path (str): Path to the CSV file with slide data.
            data_dir (str): Directory containing patch data.
            slide_dir (str): Directory containing WSI files.
            extension (str): File extension for slide images (e.g., 'tiff').
            with_data_augmentation (bool): Whether to apply data augmentation.
            max_bag_size (int): Maximum number of patches in a bag (-1 for all patches).
            is_train (bool): Whether this is a training dataset.
            is_test (bool): Whether this is a test dataset.
            slide_data (DataFrame): Slide-level metadata (e.g., slide_id, label).
            chunk_size (int): Number of patches to process at a time (used in test mode).
        """
        super().__init__(csv_path, slide_data)
        self.data_dir = data_dir
        self.slide_dir = slide_dir
        self.normalization_mean = (0.485, 0.456, 0.406)
        self.normalization_std = (0.229, 0.224, 0.225)
        self.extension = extension
        self.with_data_augmentation = with_data_augmentation
        self.max_bag_size = max_bag_size
        self.is_train = is_train
        self.is_test = is_test
        self.chunk_size = chunk_size
        self.transform = self._define_data_transforms()


    def get_init_params(self):
        """Returns parameters needed for initialization."""
        return {
            'data_dir': self.data_dir,
            'slide_dir': self.slide_dir,
            'extension': self.extension,
            'with_data_augmentation': self.with_data_augmentation,
            'max_bag_size': self.max_bag_size,
            'is_train': self.is_train,
            'is_test': self.is_test,
            'chunk_size': self.chunk_size
        }
    
    
    def _define_data_transforms(self):
        """Defines data augmentation and normalization transforms."""
        if self.with_data_augmentation and self.is_train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.01),
                transforms.ToTensor(),
                transforms.Normalize(self.normalization_mean, self.normalization_std),
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.normalization_mean, self.normalization_std),
        ])
    
    def chunk_indices(self, total, chunk_size):
        """Generates start and end indices for chunking."""
        for start in range(0, total, chunk_size):
            yield start, min(start + chunk_size, total)

    def set_range(self, start_idx, end_idx):
        """Set the range of data to be used for this dataset."""
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.current_range = (start_idx, end_idx)

    def __getitem__(self, idx):
        """Returns a bag of patches, the label, and the slide ID."""
        slide_id = self.slide_data['slide_id'].iloc[idx]
        slide_label = self.slide_data['label'].iloc[idx]
        slide_file_path = os.path.join(self.slide_dir, f"{slide_id}.{self.extension}")
        wsi = openslide.open_slide(slide_file_path)

        # Load patch coordinates and metadata
        with h5py.File(os.path.join(self.data_dir, 'patches', f"{slide_id}.h5"), 'r') as hdf5_file:
            coords = hdf5_file['coords'][:]
            patch_size = hdf5_file['coords'].attrs['patch_size']
            patch_level = hdf5_file['coords'].attrs['patch_level']

        if self.is_test:
            total_coords = len(coords)
            start, end = self.current_range

            # If range is larger than available coordinates, adjust it
            if end > total_coords:
                end = total_coords

            # Select coordinates for the current range
            selected_coords = coords[start:end]

            if total_coords <= self.current_range[0]:
                selected_coords = coords[:2]  # Mimicking previous behavior: select first two patches
        else:
            # During training, select a random subset of patches (as before)
            selected_coords = (
                coords[np.random.choice(len(coords), self.max_bag_size, replace=False)]
                if self.max_bag_size != -1 else coords
            )

        # Read patches and apply transforms
        imgs = [
            wsi.read_region((int(x), int(y)), patch_level, (patch_size, patch_size)).convert('RGB')
            for x, y in selected_coords
        ]
        return torch.stack([self.transform(img) for img in imgs]), slide_label, slide_id

