import numpy as np
import torch
from torch.utils.data import Dataset
from torch import Tensor
from typing import Optional, Tuple

class MockDataset(Dataset):
    """
    A mock dataset for testing purposes, simulating images, masks, risk factors, events, and survival times.
    This dataset generates random data to mimic the structure of a real dataset used in the STRAP project.
    
    Args: 
        n_samples (int): Number of samples in the dataset.
        image_shape (tuple): Shape of the images (channels, height, width).
        n_risk_factors (int): Number of risk factors per sample.
        seed (int, optional): Random seed for reproducibility.
    """
    def __init__(self, 
                 n_samples=100, 
                 image_shape=(1, 64, 64), 
                 n_tissues=3,
                 n_risk_factors=5, 
                 seed=None):
        
        self.n_samples = n_samples
        self.n_tissues = n_tissues
        self.image_shape = image_shape
        self.mask_shape = (self.n_tissues, *self.image_shape[1:])  # (n_tissues, height, width)
        self.n_risk_factors = n_risk_factors
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generate a random sample of data.
        Args:
            idx (int): Index of the sample to retrieve. (not used in this mock dataset)

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: A tuple containing:
                - image (Tensor): Randomly generated image tensor.
                - mask (Tensor): Randomly generated mask tensor.
                - risk_factors (Tensor): Randomly generated risk factors tensor.
                - event (Tensor): Randomly generated event indicator (0 or 1).
                - time (Tensor): Randomly generated survival time.
        """

        image           = torch.from_numpy(self.rng.normal(size=self.image_shape).astype(np.float32))
        mask            = torch.from_numpy(self.rng.integers(0, 2, size=self.mask_shape).astype(np.float32))
        risk_factors    = torch.from_numpy(self.rng.normal(size=(self.n_risk_factors,)).astype(np.float32))
        event           = torch.tensor(float(self.rng.integers(0, 2)), dtype=torch.float32)
        time            = torch.tensor(float(self.rng.uniform(0.1, 10.0)), dtype=torch.float32)
        
        return image, mask, risk_factors, event, time

# Example usage:
if __name__ == "__main__":
    dataset = MockDataset(n_samples=10)
    for i in range(len(dataset)):
        image, mask, risk_factors, event, time = dataset[i]
        print(f"Sample {i}: image shape {image.shape}, mask shape {mask.shape}, "
              f"risk_factors {risk_factors}, event {event}, time {time}")