import numpy as np
import argparse
import os
from PIL import Image
import chainer
from chainer import serializers
from chainer import Variable
from chainer import cuda
import dataset
import network
import utils
	
def generate():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', '-g', type=int, default=-1)
	parser.add_argument('--sgen', type=str, default=None)
	parser.add_argument('--depth', '-d', type=int, default=5)
	parser.add_argument('--out', '-o', type=str, default='img/')
	parser.add_argument('--num', '-n', type=int, default=100)
	args = parser.parse_args()
	
	sgen = network.StyleBasedGenerator(depth=args.depth)
	print('loading generator model from ' + args.sgen)
	serializers.load_npz(args.sgen, sgen)

	if args.gpu >= 0:
		cuda.get_device_from_id(0).use()
		sgen.to_gpu()

	xp = sgen.xp
		
	for i in range(args.num):
		print(i)
		z = sgen.make_latent(1)
		x = sgen(z, alpha=1.0)
		x = chainer.cuda.to_cpu(x.data)
		
		img = x[0].copy()
		filename = os.path.join(args.out, '%d.png' % (i + 1))
		utils.save_image(img, filename)

if __name__ == '__main__':
	generate()
