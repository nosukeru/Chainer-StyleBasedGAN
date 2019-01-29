import numpy as np
import chainer
import os
from PIL import Image

class CelebADataset(chainer.dataset.DatasetMixin):
	def __init__(self, directory, depth):
		self.directory = directory
		self.files = os.listdir(directory)
		self.depth = depth
	
	def __len__(self):
		return len(self.files)
	
	def get_example(self, i):
		img = Image.open(os.path.join(self.directory, self.files[i]))
		size = 2 ** (2 + self.depth)
		img = img.resize((size, size))
		
		img = np.array(img, dtype=np.float32) / 256
		if len(img.shape) == 2:
			img = np.broadcast_to(img, (3, img.shape[0], img.shape[1]))
		else:
			img = np.transpose(img, (2, 0, 1))

		return img
