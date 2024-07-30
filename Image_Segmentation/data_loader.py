import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt

class ImageFolder(data.Dataset):
	def __init__(self, image_paths, mask_paths, image_size=224, mode='train', augmentation_prob=0.4):
		"""Initializes image paths and preprocessing module."""
		self.image_paths = image_paths
		self.mask_paths = mask_paths
		self.image_size = image_size
		self.mode = mode
		self.RotationDegree = [0, 90, 180, 270]
		self.augmentation_prob = augmentation_prob
		print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

	def __getitem__(self, index):
		image_path = self.image_paths[index]
		mask_path = self.mask_paths[index]

		image = Image.open(image_path).convert('RGB')
		mask = Image.open(mask_path).convert('L')

		aspect_ratio = image.size[1] / image.size[0]
		Transform = []

		ResizeRange = random.randint(300, 320)
		Transform.append(T.Resize((int(ResizeRange * aspect_ratio), ResizeRange)))
		p_transform = random.random()

		if (self.mode == 'train') and p_transform <= self.augmentation_prob:
			RotationDegree = random.randint(0, 3)
			RotationDegree = self.RotationDegree[RotationDegree]
			if (RotationDegree == 90) or (RotationDegree == 270):
				aspect_ratio = 1 / aspect_ratio

			Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))

			RotationRange = random.randint(-10, 10)
			Transform.append(T.RandomRotation((RotationRange, RotationRange)))
			CropRange = random.randint(250, 270)
			Transform.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))
			Transform = T.Compose(Transform)

			image = Transform(image)
			mask = Transform(mask)

			ShiftRange_left = random.randint(0, 20)
			ShiftRange_upper = random.randint(0, 20)
			ShiftRange_right = image.size[0] - random.randint(0, 20)
			ShiftRange_lower = image.size[1] - random.randint(0, 20)
			image = image.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
			mask = mask.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))

			if random.random() < 0.5:
				image = F.hflip(image)
				mask = F.hflip(mask)

			if random.random() < 0.5:
				image = F.vflip(image)
				mask = F.vflip(mask)

			Transform = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02)

			image = Transform(image)

			Transform = []

		Transform.append(T.Resize((int(256 * aspect_ratio) - int(256 * aspect_ratio) % 16, 256)))
		Transform.append(T.Resize((self.image_size,self.image_size)))
		Transform.append(T.ToTensor())
		Transform = T.Compose(Transform)

		image = Transform(image)
		mask = Transform(mask)

		Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		image = Norm_(image)
		

		if False and (self.mode == 'train') and p_transform <= self.augmentation_prob:
			images_cpu = image.detach().cpu().numpy()
			mask_cpu = mask.detach().cpu().numpy()
			fig, axs = plt.subplots(1, 3, figsize=(15, 5))

			axs[0].imshow(np.transpose(images_cpu, (1, 2, 0)))
			axs[0].set_title('Input Image')

			axs[2].imshow(np.transpose(mask_cpu, (1, 2, 0)))
			axs[2].set_title('Ground Truth')
			plt.show()
		return image, mask, image_path


	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)

def get_loader(image_mask_pairs, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
	"""Builds and returns Dataloader."""

	image_paths, mask_paths = zip(*image_mask_pairs)
	dataset = ImageFolder(image_paths=image_paths, mask_paths=mask_paths, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
	data_loader = data.DataLoader(dataset=dataset,
								batch_size=batch_size,
								shuffle=True,
								num_workers=num_workers)
	return data_loader
