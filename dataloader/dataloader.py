import os
from random import randrange

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


single_obj_list = ['bear', 'blackswan', 'boat', 'breakdance', 'breakdance-flare', 'bus',
                   'camel', 'car-roundabout', 'car-shadow', 'car-turn', 'cows',
                   'dance-jump', 'dance-twirl', 'dog', 'drift-chicane',
                   'drift-straight', 'drift-turn', 'elephant', 'flamingo', 'goat',
                   'koala', 'libby', 'mallard-fly', 'mallard-water', 'parkour',
                   'rallye', 'rhino', 'soccerball', 'varanus-cage']


class TrainData(Dataset):
    def __init__(self, data_path, time):
        super().__init__()
        self.single_obj_list = single_obj_list
        self.data_path = data_path
        self.time = time
        self.trans = transforms.Compose([
            transforms.Resize((512, 512), interpolation=0),
            transforms.ToTensor()
            ])
        self.trans4 = transforms.Compose([
            transforms.Resize((128, 128), interpolation=0),
            transforms.ToTensor()
            ])

    def __len__(self):
        return len(single_obj_list)

    def __getitem__(self, idx):
        images = list()
        masks4 = list()
        file_names = sorted(os.listdir(self.data_path + "/JPEGImages/480p/" + self.single_obj_list[idx]))
        file_names = file_names[:int(len(file_names)*0.8)]
        randint = randrange((len(file_names)-self.time+1))
        for time in range(self.time):
            image = Image.open(
                self.data_path + "/JPEGImages/480p/" + self.single_obj_list[idx] + "/" + file_names[randint+time])
            image = self.trans(image)
            images.append(image)
            mask = Image.open(
                self.data_path + "/Annotations_unsupervised/480p/" + self.single_obj_list[idx] + "/" + file_names[randint+time][:-3]+ "png").convert('L')
            mask4 = self.trans4(mask)
            mask4 = (mask4 > 0).float()
            masks4.append(mask4)

        images = torch.stack(images, dim=0) # [t, c, h, w]
        masks4 = torch.stack(masks4, dim=1) # [c, t, h, w]
        return images, masks4


class ValidData(Dataset):
    def __init__(self, data_path, time):
        super().__init__()
        self.single_obj_list = single_obj_list
        self.data_path = data_path
        self.time = time
        self.trans = transforms.Compose([
            transforms.Resize((512, 512), interpolation=0),
            transforms.ToTensor()
            ])
        self.trans4 = transforms.Compose([
            transforms.Resize((128, 128), interpolation=0),
            transforms.ToTensor()
            ])

    def __len__(self):
        return len(single_obj_list)

    def __getitem__(self, idx):
        images = list()
        masks4 = list()
        file_names = sorted(os.listdir(self.data_path + "/JPEGImages/480p/" + self.single_obj_list[idx]))
        file_names = file_names[int(len(file_names)*0.8):]
        randint = randrange((len(file_names)-self.time+1))
        for time in range(self.time):
            image = Image.open(
                self.data_path + "/JPEGImages/480p/" + self.single_obj_list[idx] + "/" + file_names[randint+time])
            image = self.trans(image)
            images.append(image)
            mask = Image.open(
                self.data_path + "/Annotations_unsupervised/480p/" + self.single_obj_list[idx] + "/" + file_names[randint+time][:-3]+ "png").convert('L')
            mask4 = self.trans4(mask)
            mask4 = (mask4 > 0).float()
            masks4.append(mask4)

        images = torch.stack(images, dim=0) # [t, c, h, w]
        masks4 = torch.stack(masks4, dim=1) # [c, t, h, w]
        return images, masks4


class TestData(Dataset):
    def __init__(self, data_path, time):
        super().__init__()
        self.single_obj_list = single_obj_list
        self.data_path = data_path
        self.time = time
        self.trans = transforms.Compose([
            transforms.Resize((512, 512), interpolation=0),
            transforms.ToTensor()
            ])
        self.trans4 = transforms.Compose([
            transforms.Resize((128, 128), interpolation=0),
            transforms.ToTensor()
            ])

    def __len__(self):
        return len(single_obj_list)

    def __getitem__(self, idx):
        images = list()
        masks4 = list()
        file_names = sorted(os.listdir(self.data_path + "/JPEGImages/480p/" + self.single_obj_list[idx]))
        file_names = file_names[int(len(file_names)*0.8):]
        randint = randrange((len(file_names)-self.time+1))
        for time in range(self.time):
            image = Image.open(
                self.data_path + "/JPEGImages/480p/" + self.single_obj_list[idx] + "/" + file_names[randint+time])
            image = self.trans(image)
            images.append(image)

        images = torch.stack(images, dim=0) # [t, c, h, w]
        return images




def Loader(data_path, batch_size, time, num_workers, shuffle=True):
    print("Initiate DataLoader")
    train_dataset = TrainData(data_path, time)
    valid_dataset = ValidData(data_path, time)
    test_dataset = TestData(data_path, time)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=shuffle)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=shuffle)
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=1,
                              num_workers=1,
                              shuffle=False)
    data_loader = {
        "train" : train_loader,
        "valid" : valid_loader,
        "test" : test_loader
        }

    print("Train len: ", len(train_loader))
    print("Test len: ", len(test_loader))
    return data_loader

def main(idx):
    data = ValidData(data_path="./ignore/data/DAVIS", time=8)
    return data[idx]


if __name__ == '__main__':
    print('NUMBER!')
    idx = int(input())
    images, masks4 = main(idx)
    print(images.shape, masks4.shape)

