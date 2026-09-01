import os
from PIL import Image
import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils import data
from torch.utils.data import Dataset


class Dataset_loader(Dataset):

	def __init__(self, path, H=256, W=256):
		super(Dataset_loader, self).__init__()
		self.H = H
		self.W = W
		self.path = path
		self.list = os.listdir(path)
		self.transform = transforms.Compose([
			transforms.Resize((int(self.H), int(self.W))),
			# transforms.RandomCrop((self.H, self.W)),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])


	def transform_image(self, image):

		# ignore
		if image.size[0] < self.W / 2 and image.size[1] < self.H / 2:
			return None
		if image.size[0] < image.size[1] / 2 or image.size[1] < image.size[0] / 2:
			return None

		# Augment, ToTensor and Normalize
		image = self.transform(image)

		return image

	def __getitem__(self, index):

		while True:
			image = Image.open(os.path.join(self.path, self.list[index])).convert("RGB")
			image = self.transform_image(image)
			if image is not None:
				return image
			# print("dataloader : skip index", index)
			index += 1

	def __len__(self):
		return len(self.list)

class Dataset_loader_augmentation(Dataset):

    def __init__(self, path, H=256, W=256):
        super(Dataset_loader_augmentation, self).__init__()
        self.H = H
        self.W = W
        self.path = path
        self.list = os.listdir(path)

        # -------------------------
        # Transform + Augmentation
        # -------------------------
        self.transform = transforms.Compose([
            transforms.Resize((int(self.H), int(self.W))),

            #Augmentations
            transforms.RandomHorizontalFlip(p=0.5),  # left-right flip
            transforms.RandomVerticalFlip(p=0.5),    # top-bottom flip

            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def transform_image(self, image):

        # ignore very small or extreme aspect-ratio images
        if image.size[0] < self.W / 2 and image.size[1] < self.H / 2:
            return None
        if image.size[0] < image.size[1] / 2 or image.size[1] < image.size[0] / 2:
            return None

        # Apply augmentation + tensor + normalize
        image = self.transform(image)
        return image

    def __getitem__(self, index):

        while True:
            image = Image.open(
                os.path.join(self.path, self.list[index])
            ).convert("RGB")

            image = self.transform_image(image)
            if image is not None:
                return image

            index += 1

    def __len__(self):
        return len(self.list)
