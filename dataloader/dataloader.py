import torch
from torch.utils.data import Dataset, DataLoader


class TrainData(Dataset):
    def __init__(self, data_path):
        '''
        TODO:
        *** Add DAVIS 2019 unsuperrvised dataset
        * Add the ability to combine datasets
        '''
        super().__init__()
        self.path = data_path

    def __len__(self):
        '''
        TODO:
        *** Choice between and realize: (i want 2rd variant)
            1) idx --> Video(folder) number, and one random sequence for 3D input
            2) idx --> Combine all possible sequences
        '''
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

