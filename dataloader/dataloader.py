import os
from random import randrange

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torch._six import container_abcs, string_classes, int_classes


class DAVIS_train(Dataset):
    def __init__(self, time, size):
        super().__init__()
        self.data_path = "./ignore/data/DAVIS/"
        self.time = time
        with open(self.data_path + "ImageSets/2017/train.txt", 'r') as txt:
            self.video_names = txt.read().split('\n')[:-1]
        self.trans = transforms.Compose([
            transforms.Resize((size, size), interpolation=0),
            transforms.ToTensor()
            ])
        self.trans4 = transforms.Compose([
            transforms.Resize((int(size/4), int(size/4)), interpolation=0),
            transforms.ToTensor()
            ])

    def _randseq_(self, file_names, idx):
        randint = randrange((len(file_names)-self.time+1))
        images = list()
        masks = list()
        masks4emb = list()
        for time in range(self.time):
            image = self.trans(Image.open(self.data_path + "JPEGImages/480p/" + self.video_names[idx] \
                                          + "/" + file_names[randint+time]))
            mask = self.trans4(Image.open(self.data_path + "/Annotations_unsupervised/480p/" + self.video_names[idx] \
                                          + "/" + file_names[randint+time][:-3] + "png").convert('RGB'))
            mask4emb = self.trans4(Image.open(self.data_path + "/Annotations_unsupervised/480p/" + self.video_names[idx] \
                                          + "/" + file_names[randint+time][:-3] + "png").convert('L'))
            images.append(image)
            masks.append(mask)
            masks4emb.append(mask4emb)
        images = torch.stack(images, dim=0) # [t, c, h, w]
        masks = torch.stack(masks, dim=0).permute(0, 2, 3, 1) # [t, h, w, c]
        masks4emb = (torch.stack(masks4emb, dim=0).permute(0, 2, 3, 1)!=0).float()
        return images, masks, masks4emb

    def _rgb2label_(self, masks):
        m = torch.tensor([1., 10., 100.])
        labels = masks.reshape(-1, 3).unique(dim=0)
        labels = (labels*m).mean(dim=1)
        masks = (masks*m).mean(dim=3)
        new_masks = torch.zeros(masks.size(0), masks.size(1), masks.size(2), 1).int()
        for l in range(labels.size(0)):      
            new_masks += ((masks.eq(labels[l])).unsqueeze(3)*l).int()
        return new_masks

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        file_names = os.listdir(self.data_path + "JPEGImages/480p/" + self.video_names[idx])
        images, masks, masks4emb = self._randseq_(file_names, idx)
        masks = self._rgb2label_(masks) # [t, h, w, c]
        return images, masks, masks4emb


class DAVIS_valid(Dataset):
    def __init__(self, time, size):
        super().__init__()
        self.data_path = "./ignore/data/DAVIS/"
        self.time = time
        with open(self.data_path + "ImageSets/2017/val.txt", 'r') as txt:
            self.video_names = txt.read().split('\n')[:-1]
        self.trans = transforms.Compose([
            transforms.Resize((size, size), interpolation=0),
            transforms.ToTensor()
            ])
        self.trans4 = transforms.Compose([
            transforms.Resize((int(size/4), int(size/4)), interpolation=0),
            transforms.ToTensor()
            ])

    def _randseq_(self, file_names, idx):
        randint = randrange((len(file_names)-self.time+1))
        images = list()
        masks = list()
        for time in range(self.time):
            image = self.trans(Image.open(self.data_path + "JPEGImages/480p/" + self.video_names[idx] \
                                          + "/" + file_names[randint+time]))
            mask = self.trans4(Image.open(self.data_path + "/Annotations_unsupervised/480p/" + self.video_names[idx] \
                                          + "/" + file_names[randint+time][:-3] + "png").convert('RGB'))
            images.append(image)
            masks.append(mask)
        images = torch.stack(images, dim=0) # [t, c, h, w]
        masks = torch.stack(masks, dim=0).permute(0, 2, 3, 1) # [t, h, w, c]
        return images, masks

    def _rgb2label_(self, masks):
        m = torch.tensor([1., 10., 100.])
        labels = masks.reshape(-1, 3).unique(dim=0)
        labels = (labels*m).mean(dim=1)
        masks = (masks*m).mean(dim=3)
        new_masks = torch.zeros(masks.size(0), masks.size(1), masks.size(2), 1).int()
        for l in range(labels.size(0)):      
            new_masks += ((masks.eq(labels[l])).unsqueeze(3)*l).int()
        return new_masks

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        file_names = os.listdir(self.data_path + "JPEGImages/480p/" + self.video_names[idx])
        images, masks = self._randseq_(file_names, idx)
        masks = self._rgb2label_(masks) # [t, h, w, c]
        return images, masks


def Loader(size, batch_size, time, num_workers, shuffle=True):
    print("Initiate DataLoader")
    train_dataset = DAVIS_train(time=time, size=size)
    valid_dataset = DAVIS_valid(time=time, size=size)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=shuffle)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=shuffle)
    test_loader = DataLoader(dataset=valid_dataset,
                              batch_size=1,
                              num_workers=1,
                              shuffle=False)
    data_loader = {
        "train" : train_loader,
        "valid" : valid_loader,
        "test" : test_loader
        }

    print("Train len: ", len(train_loader))
    print("Valid len: ", len(valid_loader))
    print("Test len: ", len(test_loader))
    return data_loader

if __name__ == "__main__":
    data_loader = Loader(size=512, batch_size=1, time=8, num_workers=8)
    valid_dataset = DAVIS_valid(time=8, size=512)
    valid_dataset[10]
