import os

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class TrainData(Dataset):
    def __init__(self, data_path, time):
        '''
        TODO:
        *** Add DAVIS 2019 unsuperrvised dataset (i have singe video with bear)
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
        self.trans4 = transforms.Compose([
            transforms.Resize((64, 64), interpolation=0),
            transforms.ToTensor()
            ])

    def __len__(self):
        '''
        TODO:
        *** Choice between and realize: (i want 2rd variant)
            1) idx --> Video(folder) number, and one random sequence for 3D input
            2) idx --> Combine all possible sequences
        '''
        # return len(self.file_names) - self.time + 1
        return 1

    def __getitem__(self, idx):
        '''
        TODO:
        * If image not RGB, do --> .convert('RGB')
        '''
        image = Image.open(self.data_path + "/JPEGImages/480p/bear/" + self.file_names[idx])
        image = self.trans(image)
        mask = Image.open(self.data_path + "/Annotations_unsupervised/480p/bear/" + self.file_names[idx][:-3] + "png").convert('L')

        mask4 = self.trans4(mask)
        mask4 = (mask4 > 0).float()

        mask = self.trans(mask)
        mask = (mask > 0).float()
        return image, mask, mask4

def Loader(data_path, batch_size, time, num_workers, shuffle=True):
    print("Initiate DataLoader")
    train_dataset = TrainData(data_path, time)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=shuffle)
    print("Iters in epoch: ", len(train_loader))
    return train_loader

def main(idx):
    data = TrainData(data_path="./ignore/data/DAVIS", time=1)
    return data[idx]


if __name__ == '__main__':
    print('NUMBER!')
    idx = int(input())
    images, masks, masks4 = main(idx)
    print(images.shape, masks.shape, masks4.shape)

