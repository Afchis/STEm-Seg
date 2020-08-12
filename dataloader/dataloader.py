import os

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class TrainData(Dataset):
    def __init__(self, data_path, time):
        '''
        TODO:
        *** Add DAVIS 2019 unsuperrvised dataset
        * Add the ability to combine datasets
        '''
        super().__init__()
        self.data_path = data_path
        self.time = time
        self.file_names = sorted(os.listdir(data_path + "/JPEGImages/480p/bear/"))
        self.trans = transforms.Compose([
            transforms.Resize((256, 256), interpolation=0),
            transforms.ToTensor()
            ])

    def __len__(self):
        '''
        TODO:
        *** Choice between and realize: (i want 2rd variant)
            1) idx --> Video(folder) number, and one random sequence for 3D input
            2) idx --> Combine all possible sequences
        '''
        raise len(self.file_names) - self.time + 1

    def __getitem__(self, idx):
        print(self.file_names)
        '''
        TODO:
        * If image not RGB, do --> .convert('RGB')
        '''
        images = list()
        masks = list()
        for time in range(self.time):
            image = Image.open(self.data_path + "/JPEGImages/480p/bear/" + self.file_names[idx+time])
            image = self.trans(image)
            images.append(image)
            mask = Image.open(self.data_path + "/Annotations_unsupervised/480p/bear/" + self.file_names[idx+time][:-3] + "png").convert('L')
            mask = self.trans(mask)
            mask = (mask > 0).float()
            masks.append(mask)
        images = torch.stack(images, dim=0) # [t, c, h, w]
        masks = torch.stack(masks, dim=0) # [t, c, h, w]
        return images, masks


def main(idx):
    data = TrainData(data_path="./ignore/data/DAVIS", time=5)
    return data[idx]


if __name__ == '__main__':
    print('NUMBER!')
    idx = int(input())
    images, masks = main(idx)
    print(images.shape, masks.shape)

