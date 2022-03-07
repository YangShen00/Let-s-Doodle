import os
import gdown
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

DATASET_DRIVE_URL = "https://drive.google.com/drive/folders/1RJ59uDHVOlruHypJMRLoZvUQE6yD6HOM?usp=sharing"


def download_data():
    """ loads saved dataset from google drive
    """
    name = f'./npy'
    if not os.path.exists(name):
        os.makedirs(name)
        gdown.download_folder(DATASET_DRIVE_URL, quiet=True)
        
class QuickDrawDataset(Dataset):
    """QuickDraw Dataset, data sourced from Google."""
    
    def __init__(self, npy_directory="./npy", transform=None, prop=0.10):
        """
        Args:
            npy_directory (string): Path to the npy directory with sketch information.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if npy_directory is None:
            download_data()
            npy_directory = f'./npy'
            
        height, width = 28, 28
        encoding_count = 0
        self.encoding = {}
        self.sketches = np.empty([0, height, width])
        self.labels = np.empty([0])
        
        for file in os.listdir(npy_directory):
            filename = os.fsdecode(file)
            if filename.endswith("npy"):
                self.encoding[encoding_count] = filename
                all_imgs = np.load(os.path.join(npy_directory, filename))
                count, _ = all_imgs.shape
                sample_count = int(count*prop)
                sample_imgs = all_imgs[:sample_count].reshape((sample_count, height, width))
                self.sketches = np.concatenate((self.sketches, sample_imgs))
                self.labels = np.concatenate((self.labels, np.full((sample_count), encoding_count)))
                encoding_count += 1
            else:
                continue
         
        self.transform = transform

    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = torch.from_numpy(self.sketches[idx])
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label
    