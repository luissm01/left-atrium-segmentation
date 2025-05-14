from pathlib import Path

import torch
import numpy as np
import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

class CardiacDataset(torch.utils.data.Dataset):
    def __init__(self, root, augment_params):
        # Constructor: receives the dataset root (train or val) and the augmentation parameters
        self.all_files = self.extract_files(root)  # List of paths to all slices
        self.augment_params = augment_params       # Augmentation parameters
    
    @staticmethod
    def extract_files(root):
        """
        Extracts the paths to all slices (.npy files) under the given root directory.
        (Root should point to either train or val directory.)
        
        @staticmethod is used because this method does not depend on the instance (self) 
        or the class itself. It only needs the 'root' argument.
        """
        files = []
        for subject in root.glob("*"):   # Iterate over all subjects (one folder per patient)
            slice_path = subject/"slices"  # Access the 'slices' folder for each subject
            for slice in slice_path.glob("*.npy"):  # Find all .npy slice files
                files.append(slice)     # Append the path to the list
        return files
    
    @staticmethod
    def change_img_to_label_path(path):
        """
        Given a path to a slice, replace 'slices' with 'masks' to get the corresponding mask path.
        
        @staticmethod is used because this function simply transforms a given path
        and does not require access to the class or instance attributes.
        """
        parts = list(path.parts)  # Split the path into parts
        parts[parts.index("slices")] = "masks"  # Replace "slices" with "masks" in the path
        return Path(*parts)       # Reconstruct the path

    def augment(self, slice, mask):
        """
        Applies the same augmentation to both the slice and the corresponding mask.
        A manual seed is initialized to ensure that different DataLoader workers 
        do not generate the same random augmentations (common PyTorch issue).
        """
        random_seed = torch.randint(0, 1000000, (1,)).item()  # Generate a random seed
        imgaug.seed(random_seed)  # Set the seed for imgaug to ensure randomness
        
        mask = SegmentationMapsOnImage(mask, mask.shape)  # Wrap the mask for imgaug
        slice_aug, mask_aug = self.augment_params(image=slice, segmentation_maps=mask)  # Apply augmentation
        mask_aug = mask_aug.get_arr()  # Convert back to a NumPy array
        return slice_aug, mask_aug

    def __len__(self):
        """
        Returns the total number of slices available in the dataset.
        Required for PyTorch's DataLoader to work properly.
        """
        return len(self.all_files)
    
    def __getitem__(self, idx):
        """
        Given an index, returns the corresponding (augmented) slice and its mask.
        
        - Loads the slice and mask from .npy files.
        - Applies augmentations if they are provided.
        - Adds an extra channel dimension (C, H, W) since the original slices are (H, W),
          and PyTorch expects inputs of shape (C, H, W).
        """
        file_path = self.all_files[idx]  # Path to the slice
        mask_path = self.change_img_to_label_path(file_path)  # Path to the corresponding mask
        
        slice = np.load(file_path).astype(np.float32)  # Load slice and convert to float32
        mask = np.load(mask_path)  # Load mask
        
        if self.augment_params:
            slice, mask = self.augment(slice, mask)  # Apply augmentation if needed
        
        return np.expand_dims(slice, 0), np.expand_dims(mask, 0)  # Add channel dimension (C, H, W)
        