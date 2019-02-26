import numpy as np
import argparse
import chainer
from chainer import training
from chainer import iterators, optimizers, serializers
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import dataset
import network
import utils
from updater import R1Updater
import os
import shutil

def train():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', '-g', type=int, default=-1)
	parser.add_argument('--dir', type=str, default='train_celeba/')
	parser.add_argument('--enc', type=str, default=None)
	parser.add_argument('--sgen', type=str, default='./sgen')
	parser.add_argument('--dis', type=str, default='./dis')
	parser.add_argument('--opte', type=str, default=None)
	parser.add_argument('--epoch', '-e', type=int, default=2)
	parser.add_argument('--lr', '-l', type=float, default=0.001)
	parser.add_argument('--beta1', type=float, default=0)
	parser.add_argument('--beta2', type=float, default=0.99)
	parser.add_argument('--batch', '-b', type=int, default=16)
	parser.add_argument('--depth', '-d', type=int, default=0)
	parser.add_argument('--out', '-o', type=str, default='img/')
	parser.add_argument('--num', '-n', type=int, default=10)
	args = parser.parse_args()

	train = dataset.CelebADataset(directory=args.dir, depth=args.depth)
	train_iter = iterators.SerialIterator(train, batch_size=args.batch)

	enc = network.ConvEncoder()
	if args.enc is not None:
		print('loading conv-encoder model from ' + args.enc)
		serializers.load_npz(args.enc, enc)
	
	sgen = network.StyleBasedGenerator(depth=args.depth)
	print('loading style-based-generator model from ' + args.sgen)
	serializers.load_npz(args.sgen, sgen)
	
	dis = network.Discriminator(depth=args.depth)
	print('loading discriminator model from ' + args.dis)
	serializers.load_npz(args.dis, dis)
			
	if args.gpu >= 0:
		cuda.get_device_from_id(args.gpu).use()
		enc.to_gpu()
		sgen.to_gpu()
		dis.to_gpu()

	opt_e = optimizers.Adam(alpha=args.lr, beta1=args.beta1, beta2=args.beta2)
	opt_e.setup(enc)
	if args.opte is not None:
		print('loading conv-encoder optimizer from ' + args.opte)
		serializers.load_npz(args.opte, opt_e)
	
	updater = AEUpdater(models=(enc, sgen, dis),
						iterator={'main': train_iter},
						optimizer={'enc': opt_e},
						device=args.gpu)

	trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='results')

	if os.path.isdir(args.out):
		shutil.rmtree(args.out)
	os.makedirs(args.out)
	
	def output_image(enc, sgen, train, depth, out, num):
		@chainer.training.make_extension()
		def make_image(trainer):
			x = train[np.random.randint(len(train), size=num)]
			y = sgen.G(enc(x), 1.0)
			y = chainer.cuda.to_cpu(y.data)

			for i in range(num):
				img = y[i].copy()
				filename = os.path.join(out, '%d_%d.png' % (trainer.updater.iteration, i))
				utils.save_image(img, filename)

		return make_image
			
	trainer.extend(extensions.LogReport(trigger=(1000, 'iteration')))
	trainer.extend(extensions.PrintReport(['iteration', 'alpha', 'loss_feat', 'loss_lat']))
	trainer.extend(output_image(enc, sgen, train, args.depth, args.out, args.num), trigger=(1000, 'iteration'))
	trainer.extend(extensions.ProgressBar(update_interval=1))	
	
	trainer.run()

	modelname = './results/enc'
	print('saving conv-encoder model to ' + modelname)
	serializers.save_npz(modelname, enc)

	optname = './results/opt_e'
	print('saving style-encoder optimizer to ' + optname)
	serializers.save_npz(optname, opt_e)

if __name__ == '__main__':
	train()
