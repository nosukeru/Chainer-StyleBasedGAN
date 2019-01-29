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
	parser.add_argument('--sgen', type=str, default=None)
	parser.add_argument('--dis', type=str, default=None)
	parser.add_argument('--optg', type=str, default=None)
	parser.add_argument('--opte', type=str, default=None)
	parser.add_argument('--optd', type=str, default=None)
	parser.add_argument('--epoch', '-e', type=int, default=3)
	parser.add_argument('--lr', '-l', type=float, default=0.001)
	parser.add_argument('--beta1', type=float, default=0)
	parser.add_argument('--beta2', type=float, default=0.99)
	parser.add_argument('--batch', '-b', type=int, default=16)
	parser.add_argument('--depth', '-d', type=int, default=0)
	parser.add_argument('--alpha', type=float, default=0)
	parser.add_argument('--delta', type=float, default=0.00005)
	parser.add_argument('--out', '-o', type=str, default='img/')
	parser.add_argument('--num', '-n', type=int, default=10)
	args = parser.parse_args()

	train = dataset.CelebADataset(directory=args.dir, depth=args.depth)
	train_iter = iterators.SerialIterator(train, batch_size=args.batch)

	sgen = network.StyleBasedGenerator(depth=args.depth)
	if args.sgen is not None:
		print('loading style-based-generator model from ' + args.sgen)
		serializers.load_npz(args.sgen, sgen)
	
	dis = network.Discriminator(depth=args.depth)
	if args.dis is not None:
		print('loading discriminator model from ' + args.dis)
		serializers.load_npz(args.dis, dis)
			
	if args.gpu >= 0:
		cuda.get_device_from_id(args.gpu).use()
		sgen.to_gpu()
		dis.to_gpu()

	opt_g = optimizers.Adam(alpha=args.lr, beta1=args.beta1, beta2=args.beta2)
	opt_g.setup(sgen.G)
	if args.optg is not None:
		print('loading generator optimizer from ' + args.optg)
		serializers.load_npz(args.optg, opt_g)

	opt_e = optimizers.Adam(alpha=args.lr * 0.01, beta1=args.beta1, beta2=args.beta2)
	opt_e.setup(sgen.E)
	if args.opte is not None:
		print('loading style-encoder optimizer from ' + args.opte)
		serializers.load_npz(args.opte, opt_e)
	
	opt_d = optimizers.Adam(alpha=args.lr, beta1=args.beta1, beta2=args.beta2)
	opt_d.setup(dis)
	if args.optd is not None:
		print('loading discriminator optimizer from ' + args.optd)
		serializers.load_npz(args.optd, opt_d)

	updater = R1Updater(alpha=args.alpha,
							delta=args.delta,
							models=(sgen, dis),
							iterator={'main': train_iter},
							optimizer={'gen': opt_g, 'enc': opt_e, 'dis': opt_d},
							device=args.gpu)

	trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='results')

	if os.path.isdir(args.out):
		shutil.rmtree(args.out)
	os.makedirs(args.out)
	for i in range(args.num):
		img = train.get_example(i)
		filename = os.path.join(args.out, 'real_%d.png' % i)
		utils.save_image(img, filename)
	
	def output_image(sgen, depth, out, num):
		@chainer.training.make_extension()
		def make_image(trainer):
			z = sgen.make_latent(num)
			x = sgen(z, alpha=trainer.updater.alpha)
			x = chainer.cuda.to_cpu(x.data)

			for i in range(args.num):
				img = x[i].copy()
				filename = os.path.join(out, '%d_%d.png' % (trainer.updater.iteration, i))
				utils.save_image(img, filename)

		return make_image
			
	trainer.extend(extensions.LogReport(trigger=(1000, 'iteration')))
	trainer.extend(extensions.PrintReport(['iteration', 'alpha', 'loss_gen', 'loss_dis']))
#	trainer.extend(extensions.PrintReport(['iteration', 'alpha', 'gen_loss', 'dis_loss', 'loss_d', 'loss_l', 'loss_dr']))
	trainer.extend(output_image(sgen, args.depth, args.out, args.num), trigger=(1000, 'iteration'))
	trainer.extend(extensions.ProgressBar(update_interval=1))	
	
	trainer.run()

	modelname = './results/sgen'
	print( 'saving style-based-generator model to ' + modelname )
	serializers.save_npz(modelname, sgen)

	modelname = './results/dis'
	print( 'saving discriminator model to ' + modelname )
	serializers.save_npz(modelname, dis)

	optname = './results/opt_g'
	print( 'saving generator optimizer to ' + optname )
	serializers.save_npz(optname, opt_g)

	optname = './results/opt_e'
	print( 'saving style-encoder optimizer to ' + optname )
	serializers.save_npz(optname, opt_e)

	optname = './results/opt_d'
	print( 'saving generator optimizer to ' + optname )
	serializers.save_npz(optname, opt_d)

if __name__ == '__main__':
	train()
