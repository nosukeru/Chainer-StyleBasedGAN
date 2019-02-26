import numpy as np
import chainer
import chainer.functions as F
from chainer import reporter
from chainer import Variable

class R1Updater(chainer.training.StandardUpdater):
	def __init__(self, alpha, delta, **kwargs):
		self.sgen, self.dis = kwargs.pop('models')
		self.gamma = 10
		self.alpha = alpha
		self.delta = delta
		super(R1Updater, self).__init__(**kwargs)

	def update_core(self):
		opt_g = self.get_optimizer('gen')
		opt_e = self.get_optimizer('enc')
		opt_d = self.get_optimizer('dis')

		xp = self.sgen.xp

		# train discriminator
		batch = self.get_iterator('main').next()
		batchsize = len(batch)

		x_real = Variable(xp.array(batch))
		y_real = self.dis(x_real, self.alpha)

		g_d, = chainer.grad([y_real], [x_real], enable_double_backprop=True)
		g_d_norm = F.sum(F.batch_l2_norm_squared(g_d)) / batchsize
		loss_gp = self.gamma * g_d_norm / 2

		z = Variable(self.sgen.make_latent(batchsize))
		x_fake = self.sgen(z, self.alpha)
		y_fake = self.dis(x_fake, self.alpha)

		loss_dis = F.sum(F.softplus(-y_real)) / batchsize
		loss_dis += F.sum(F.softplus(y_fake)) / batchsize
		loss_dis += loss_gp

		x_fake.unchain_backward()
		self.dis.cleargrads()
		loss_dis.backward()
		opt_d.update()

		# train generator
		z = Variable(self.sgen.make_latent(batchsize))
		x_fake = self.sgen(z, self.alpha)
		y_fake = self.dis(x_fake, self.alpha)

		loss_gen = F.sum(F.softplus(-y_fake)) / batchsize

		self.sgen.cleargrads()
		loss_gen.backward()
		opt_g.update()
		opt_e.update()

		reporter.report({'alpha': self.alpha})
		reporter.report({'loss_gen': loss_gen})
		reporter.report({'loss_dis': loss_dis})
				
		self.alpha = self.alpha + self.delta


class WganGpUpdater(chainer.training.StandardUpdater):
	def __init__(self, alpha, delta, **kwargs):
		self.sgen, self.dis = kwargs.pop('models')
		self.lam = 10
		self.epsilon_drift = 0.001
		self.alpha = alpha
		self.delta = delta
		super(WganGpUpdater, self).__init__(**kwargs)

	def update_core(self):
		opt_g = self.get_optimizer('gen')
		opt_e = self.get_optimizer('enc')
		opt_d = self.get_optimizer('dis')

		xp = self.sgen.xp

		# update discriminator
		x = self.get_iterator('main').next()
		x = xp.array(x)
		m = len(x)

		z = self.sgen.make_latent(m)
		x_tilde = self.sgen(z, self.alpha).data
		
		epsilon = xp.random.rand(m, 1, 1, 1).astype('f')
		x_hat = Variable(epsilon * x + (1 - epsilon) * x_tilde)

		dis_x = self.dis(x, self.alpha)
		
		loss_d = self.dis(x_tilde, self.alpha) - dis_x

		g_d, = chainer.grad([self.dis(x_hat, self.alpha)], [x_hat], enable_double_backprop=True)
		g_d_norm = F.sqrt(F.batch_l2_norm_squared(g_d) + 1e-6)
		g_d_norm_delta = g_d_norm - 1
		loss_l = self.lam * g_d_norm_delta * g_d_norm_delta
		
		loss_dr = self.epsilon_drift * dis_x * dis_x

		dis_loss = F.mean(loss_d + loss_l + loss_dr)

		self.dis.cleargrads()
		dis_loss.backward()
		opt_d.update()
		
		# update generator
		z = self.sgen.make_latent(m)
		x = self.sgen(z, self.alpha)
		gen_loss = F.average(-self.dis(x, self.alpha))

		self.sgen.cleargrads()
		gen_loss.backward()
		opt_g.update()
		opt_e.update()

		reporter.report({'loss_d': F.mean(loss_d), 'loss_l': F.mean(loss_l), 'loss_dr': F.mean(loss_dr), 'dis_loss': dis_loss, 'gen_loss': gen_loss, 'alpha': self.alpha})

		self.alpha = self.alpha + self.delta


class AEUpdater(chainer.training.StandardUpdater):
	def __init__(self, **kwargs):
		self.enc, self.sgen, self.dis = kwargs.pop('models')
		super(AEUpdater, self).__init__(**kwargs)

	def update_core(self):
		opt_e = self.get_optimizer('enc')

		xp = self.sgen.xp

		# train encoder
		batch = self.get_iterator('main').next()
		batchsize = len(batch)

		x_real = Variable(xp.array(batch))
		w_rec = self.enc(x_real)
		x_rec = self.sgen.G(w_rec, 1.0)
		y_rec = self.dis(x_rec, 1.0)

		loss_gen = F.mean_absolute_error(x_rec, x_real)
		loss_dis = F.sum(F.softplus(-y_rec)) / batchsize

		loss = loss_gen + loss_dis
		self.enc.cleargrads()
		loss.backward()
		opt_e.update()

		reporter.report({'loss_gen': loss_gen})
		reporter.report({'loss_dis': loss_dis})

