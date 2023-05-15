import os
import numpy as np

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
        data = np.load(os.path.join(self.file_dir,file))
        label = int(file[-5:-4])
        # if len(data.shape) == 2:
        #     data = np.expand_dims(data, axis=2)
        # if self.transform is not None:
        #     data = self.transform(np.float32(data))
            #data = torch.squeeze(data)
        data = torch.Tensor(np.float32(data.transpose(1,0)))
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
        label = int(file[-5:-4])
        # if len(data.shape) == 2:
        #     data = np.expand_dims(data, axis=2)
            #data = data.transpose((2, 1, 0))
        # if self.transform is not None:
        #     data = self.transform(np.float32(data))
        data = torch.Tensor(np.float32(data.transpose(1, 0)))
        return data, label

    def __len__(self):
        return len(self.files)



class FCNDataset(Dataset):
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
        label = int(file[6:8])

        data = data.transpose((1, 0))
        if self.transform is not None:
            data = self.transform(np.float32(data))
            data = torch.squeeze(data)

        return data, label

    def __len__(self):
        return len(self.files)

class FCNDatasetList(Dataset):
    def __init__(self, data, label, transform=None):
        self.transform = transform
        self.data_list = data
        self.label_list = label

    def __getitem__(self, i):
        data = self.data_list[i]
        data = data.transpose((1, 0))

        label = int(self.label_list)

        if self.transform is not None:
            data = self.transform(np.float32(data))
            data = torch.squeeze(data)

        return data, label

    def __len__(self):
        return len(self.data_list)