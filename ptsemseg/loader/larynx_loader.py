import os
import torch
import numpy as np
import cv2

from torch.utils import data

from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import *


class larynxLoader(data.Dataset):

	def __init__(self, split="train", is_transform=False, img_size=(480, 640), augmentations=None, img_norm=True):
		"""__init__
		:param root:
		:param split:
		:param is_transform:
		:param img_size:
		:param augmentations
		"""

		self.root = "/home/felix/projects/larynx/data/"
		self.split = split
		self.is_transform = is_transform
		self.augmentations = augmentations
		self.img_norm = img_norm
		self.n_classes = 3
		self.img_size = img_size if isinstance(img_size, tuple) or isinstance(img_size, list) else (img_size, img_size)
		self.mean = np.array([103.939, 116.779, 123.68])
		self.ignore_index = 250
		self.files = {}

		self.void_classes = []
		self.valid_classes = [0, 1, 2]
		self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))

		self.colors = [
			[  0,   0,   0],
			[255,   0,   0],
			[  0,   0, 255],
		]
		self.label_colours = dict(zip(range(self.n_classes), self.colors))

		self.images_base = os.path.join(self.root, self.split, "images")
		self.annotations_base = os.path.join(self.root, self.split, "annotations")
		self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

		if not self.files[split]:
			raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

		print("Found %d %s images" % (len(self.files[split]), split))

	def __len__(self):
		"""__len__"""
		return len(self.files[self.split])

	def __getitem__(self, index):
		"""__getitem__
		:param index:
		"""

		img_path = self.files[self.split][index].rstrip()
		lbl_path = os.path.join(self.annotations_base, os.path.basename(img_path))

		img = cv2.imread(img_path)
		lbl = cv2.imread(lbl_path, 0)

		lbl = self.encode_segmap(lbl)

		if self.augmentations is not None:
			img, lbl = self.augmentations(img, lbl)

		if self.is_transform:
			img, lbl = self.transform(img, lbl)

		return img, lbl

	def transform(self, img, lbl):
		"""transform

		:param img:
		:param lbl:
		"""

		img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_AREA)

		img = img.astype(np.float64)
		img -= self.mean
		img = img / 255.0
		img = img.transpose(2, 0, 1) # NHWC -> NCHW

		classes = np.unique(lbl)
		lbl = cv2.resize(lbl, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

		if not np.all(classes == np.unique(lbl)):
			print("WARN: resizing labels yielded fewer classes")

		if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
			print("after det", classes, np.unique(lbl))
			raise ValueError("Segmentation map contained invalid class values")

		img = torch.from_numpy(img).float()
		lbl = torch.from_numpy(lbl).long()

		return img, lbl

	def encode_segmap(self, mask):
		for _voidc in self.void_classes:
			mask[mask == _voidc] = self.ignore_index
		for _validc in self.valid_classes:
			mask[mask == _validc] = self.class_map[_validc]
		return mask

	def decode_segmap(self, temp):
		dest = np.zeros((temp.shape[0], temp.shape[1], 3))
		for l in range(0, self.n_classes):
			dest[temp == l] = self.label_colours[l]
		return dest

	def decode_image(self, img):
		img = img.transpose(1, 2, 0) # NHWC -> NCHW
		img = img.astype(np.float64)
		img = img * 255.0
		img += self.mean
		img = img.astype(np.uint8)
		return img