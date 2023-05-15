import os
import numpy as np
import h5py
import torch
from PIL import Image
from torch.utils.data import Dataset

class MetaDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.train = train
        self.transform = transform
        self.training_file = 'training'
        self.validation_file = 'validation'

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.validation_file

        self.file_dir = os.path.join(root, data_file)

        self.files = [file for file in os.listdir(self.file_dir) if not file.startswith('.')]

    def __getitem__(self, i):
        file = self.files[i]
        data = np.load(os.path.join(self.file_dir, file))
        data = data.astype(np.float32)
        data = data.transpose(3, 0, 1, 2)
        label = int(file[-5:-4])
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=2)

        return data, label

    def __len__(self):
        return len(self.files)



class DatasetList(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.file_dir = root
        self.files = [file for file in os.listdir(self.file_dir) if not file.startswith('.')]


    def __getitem__(self, i):
        file = self.files[i]
        data = np.load(os.path.join(self.file_dir, file))
        data = data.astype(np.float32)
        data = data.transpose(3, 0, 1, 2)
        label = int(file[-5:-4])
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=2)
            #data = data.transpose((2, 1, 0))

        # if self.transform is not None:
        #     data = self.transform(np.float32(data))

        return data, label

    def __len__(self):
        return len(self.files)