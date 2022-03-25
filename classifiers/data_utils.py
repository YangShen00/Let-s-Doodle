import os
import gdown
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pandas as pd
import ujson as json
from binary_file_parser import *

DATASET_DRIVE_URL = "https://drive.google.com/drive/folders/1RJ59uDHVOlruHypJMRLoZvUQE6yD6HOM?usp=sharing"

def download_data():
    """ loads saved dataset from google drive
    """
    name = f'./ndjson'
    if not os.path.exists(name):
        os.makedirs(name)
        gdown.download_folder(DATASET_DRIVE_URL, quiet=True)
        
def open_ndjson(path: str) -> pd.DataFrame:
    """
    Pass in input filepath to ndjson, return pandas DataFrame.
    """
    records = map(json.loads, open(path, encoding="utf8"))
    return pd.DataFrame.from_records(records)
        
def rand_train_test_idx(n, train_prop=.5, valid_prop=.25, seed=42):
    """ randomly splits label into train/valid/test splits """
    np.random.seed(seed)

    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    return train_indices, val_indices, test_indices

def load_npy(file_path):
    rasters = None
    with open(file_path, 'rb') as f:
        rasters = np.load(f)

    return rasters
        
        
class QuickDrawDataset(Dataset):
    """QuickDraw Dataset, data sourced from Google."""
    
    def __init__(self, ndjson_directory="./ndjson", transform=None, prop=0.10):
        """
        Args:
            ndjson_directory (string): Path to the ndjson directory with sketch information.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if ndjson_directory is None:
            download_data()
            ndjson_directory = f'./ndjson'
            
        height, width = 28, 28
        encoding_count = 0
        self.encoding = {}
        self.sketches = np.empty([0, height, width])
        self.labels = []
        
        for file in tqdm(os.listdir(ndjson_directory)):
            filename = os.fsdecode(file)
            if filename.endswith("ndjson"):
                # retaining only images that have been recognized
                df = open_ndjson(os.path.join(ndjson_directory, filename))
                df = df.loc[df.recognized, :]
                
                # creating one hot encoding for the string label
                self.encoding[encoding_count] = df.word.unique()[0]
                
                sample_count = int(len(df)*prop)
                
                cached_file = f"rasters/{self.encoding[encoding_count]}.npy"
                if not os.path.exists(cached_file):
                    print(f"Generating raster data for {self.encoding[encoding_count]}.")
                    sample_imgs = vector_to_raster(df.drawing[:sample_count], self.encoding[encoding_count])

                sample_imgs = load_npy(cached_file)
                sample_imgs = sample_imgs.reshape((sample_count, height, width))
                assert sample_imgs.shape[0] == sample_count, 'Wrong number of rasters generated'
                
                # append array of sampled sketches to all sketches
                self.sketches = np.concatenate((self.sketches, sample_imgs))
                
                print("Category: {}. Finished processing {} images"
                      .format(self.encoding[encoding_count], sample_imgs.shape[0]))

                self.labels = self.labels + [encoding_count] * sample_count
                encoding_count += 1
            else:
                continue

        self.labels = torch.as_tensor(self.labels)
        
        self.transform = transform
    
    def get_idx_split(self, train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """
        train_idx, valid_idx, test_idx = rand_train_test_idx(len(self.sketches), 
                                                             train_prop=train_prop, 
                                                             valid_prop=valid_prop)
        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
        return split_idx

    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = torch.from_numpy(self.sketches[idx]).float()
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)

        return img, label
    

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)

@torch.no_grad()
def evaluate(model, dataset, split_idx, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        if not sampling:
            out = model(dataset)
        else:
            out = model.inference(dataset, subgraph_loader)

    train_acc = eval_acc(
        dataset.labels[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_acc(
        dataset.labels[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_acc(
        dataset.labels[split_idx['test']], out[split_idx['test']])

    return train_acc, valid_acc, test_acc, out