import numpy as np
import chainer
import chainer.functions as F
from chainer import serializers
from PIL import Image
import network
import utils

if __name__ == '__main__':
	image = Image.open("real/real_1.png")
	y = np.array(image.getdata()).astype('float32').reshape((1, 3, 128, 128)) / 255.0
	
	sgen = network.StyleBasedGenerator(depth=5)
	serializers.load_npz("sgen", sgen)
	
	W = sgen.E(sgen.make_latent(10))
	X = sgen.G(W, alpha=1.0)
	
	nx = X[0]
	for i in range(100):
		if F.mean_squared_error(X[i], y) < F.mean_squared_error(nx, y):
			nx = X[i]

	utils.save_image(nx, "nearest.jpg")	
	
	
