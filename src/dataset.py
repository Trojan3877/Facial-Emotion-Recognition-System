"""
================================================================================
FACIAL EMOTION RECOGNITION — FER-2013 PYTORCH DATASET

Purpose:
    Provides a reusable PyTorch Dataset for loading and preprocessing
    FER-2013 pixel data from a pandas DataFrame.

Usage:
    from src.dataset import FERDataset
================================================================================
"""

import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_SIZE = 48


class FERDataset(Dataset):
    """
    Custom PyTorch Dataset for FER-2013.

    Args:
        dataframe: pandas DataFrame with 'pixels' (space-separated ints)
                   and 'emotion' (int label 0-6) columns.
    """

    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pixels = np.array(row["pixels"].split(), dtype="float32")
        image = pixels.reshape(1, IMAGE_SIZE, IMAGE_SIZE) / 255.0
        label = int(row["emotion"])

        return torch.tensor(image), torch.tensor(label)
