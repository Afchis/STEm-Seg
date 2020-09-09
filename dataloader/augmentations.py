import numpy as np
import random

from PIL import Image
import imgaug.augmenters as iaa

import torch
from torchvision import transforms


def RotateVideo(images, masks):
    degrees = [0, 90]
    choice = random.choice(degrees)
    if choice == 0:
        return images, masks
    if choice == 90:
        return images.flip(3), masks.flip(3)


class ImgsAug():
    def __init__(self, size):
        self.resize_img = transforms.Resize((size, size), interpolation=0)
        self.resize_mask = transforms.Resize((int(size/4), int(size/4)), interpolation=0)
        self.to_tensor = transforms.ToTensor()
        self.adjust_hue = 0
        self.rotate = 0

    def Rand(self):
        self.adjust_hue = random.uniform(-0.5, 0.5)
        self.rotate = random.randint(-15, 15)

    def Image(self, image):
        image = self.resize_img(image)
        image = transforms.functional.adjust_hue(image, self.adjust_hue)
        image = transforms.functional.rotate(image, self.rotate)
        image = self.to_tensor(image)
        return image

    def Mask(self, mask):
        mask = self.resize_mask(mask)
        mask = transforms.functional.rotate(mask, self.rotate)
        mask = self.to_tensor(mask)
        return mask

