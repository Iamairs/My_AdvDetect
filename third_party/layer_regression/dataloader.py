import torch
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import pandas as pd
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


IMAGENET_VAL_TXT = 'dataset/imagenet/val.txt'
IMAGENET_VAL_PATH = 'dataset/imagenet/val/'
IMAGENET_VAL_SORTED = 'dataset/imagenet/val_sorted.pkl'


def split_val():
	df = pd.read_pickle(IMAGENET_VAL_SORTED)
	image_names = df[0].tolist()
	labels = df[1].tolist()

	test_split = []
	test_split_labels = []

	train_split = []
	train_split_labels = []

	for i in range(len(image_names)):
		if i % 5 == 0:
			test_split.append(IMAGENET_VAL_PATH + image_names[i])
			test_split_labels.append(labels[i])
		else:
			train_split.append(IMAGENET_VAL_PATH + image_names[i])
			train_split_labels.append(labels[i])

	return train_split, train_split_labels, test_split, test_split_labels


class ImageNetDataset(Dataset):
	def __init__(self, split, resize_dim=224):
		self.resize_dim = resize_dim
		self.split = split

		train_split, train_split_labels, test_split, test_split_labels = split_val()
		if self.split == 'test':
			self.image_paths = test_split
			self.image_labels = test_split_labels
		if self.split == 'train':
			self.image_paths = train_split
			self.image_labels = train_split_labels

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		image = Image.open(self.image_paths[idx])
		image = pil_to_tensor(image)

		image = Resize((self.resize_dim, self.resize_dim))(image)
		label = self.image_labels[idx]

		# channel handling
		if image.shape[0] == 1:
			#print('shape expansion')
			image = image.expand(3, *image.shape[1:])
		if image.shape[0] == 4:
			new_transform = transforms.Lambda(lambda x: x[:3])
			image = new_transform(image)
		if image.shape[0] != 3:
			print(image.shape)
			print(self.image_paths[idx])

		image = image / 255

		return image, label


def get_loader(split, resize_dim=224, batch=10, shuffle=False):
	dataset = ImageNetDataset(split, resize_dim)
	loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=False)

	return loader
