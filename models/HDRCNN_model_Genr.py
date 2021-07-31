import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .losses import init_loss

try:
	xrange          # Python2
except NameError:
	xrange = range  # Python 3

class HDRCNN_Genr(BaseModel):
	def name(self):
		return 'HDRCNN_GenrModel'

	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		self.isTrain = opt.isTrain
		# define tensors
		self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
								   opt.fineSize, opt.fineSize)
		self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
								   opt.fineSize, opt.fineSize)

		# load/define networks
		#Temp Fix for nn.parallel as nn.parallel crashes oc calculating gradient penalty
		use_parallel = True
		self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
									  opt.which_model_netG,opt.segmroot, opt.norm, not opt.no_dropout, self.gpu_ids, use_parallel, opt.learn_residual)

		if not self.isTrain or opt.continue_train:
			self.load_network(self.netG, 'G', opt.which_epoch)


		if self.isTrain:
			self.fake_AB_pool = ImagePool(opt.pool_size)
			self.old_lr = opt.lr

			# initialize optimizers
			self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
												lr=opt.lr, betas=(opt.beta1, 0.999))

			# define loss functions
			# self.discLoss, self.contentLoss = init_loss(opt, self.Tensor)
			self.contentLoss = init_loss(opt, self.Tensor)

		print('---------- Networks initialized -------------')
		networks.print_network(self.netG)
		print('-----------------------------------------------')

	def set_input(self, input):
		AtoB = self.opt.which_direction == 'AtoB'
		input_A = input['A' if AtoB else 'B']
		input_B = input['B' if AtoB else 'A']
		self.input_A.resize_(input_A.size()).copy_(input_A)
		self.input_B.resize_(input_B.size()).copy_(input_B)
		#self.image_paths = input['A_paths' if AtoB else 'B_paths']

	def forward(self):
		self.real_A = Variable(self.input_A)
		self.fake_B = self.netG.forward(self.real_A)
		self.real_B = Variable(self.input_B)

	# no backprop gradients
	def test(self):
		self.real_A = Variable(self.input_A, volatile=True)
		self.fake_B = self.netG.forward(self.real_A)
		self.real_B = Variable(self.input_B, volatile=True)

	# get image paths
	def get_image_paths(self):
		return self.image_paths

	def backward_G(self):
		# Second, G(A) = B
		# self.loss_G_Content = self.contentLoss.get_loss(self.fake_B, self.real_B) * self.opt.lambda_A
		self.loss_G_Content = self.contentLoss.get_loss(self.fake_B, self.real_B)
		#self.loss_G_Percep = self.percepLoss.get_loss(self.fake_B, self.real_B)
		self.loss_G = self.loss_G_Content
		self.loss_G.backward()

	def optimize_parameters(self):
		self.forward()
		self.optimizer_G.zero_grad()
		self.backward_G()
		self.optimizer_G.step()

	def get_current_errors(self):

		return OrderedDict([('G_total', self.loss_G.item()),
							('G_L1', self.loss_G_Content.item())
							])

	def get_current_visuals(self):
		real_A = util.tensor2im(self.real_A.data)
		fake_B = util.tensor2im(self.fake_B.data)
		real_B = util.tensor2im(self.real_B.data)
		return OrderedDict([('Blurred_Train', real_A), ('Restored_Train', fake_B), ('Sharp_Train', real_B)])

	def save(self, label):
		self.save_network(self.netG, 'G', label, self.gpu_ids)

	def update_learning_rate(self):
		lrd = self.opt.lr / self.opt.niter_decay
		lr = self.old_lr - lrd
		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr
		print('update learning rate: %f -> %f' % (self.old_lr, lr))
		self.old_lr = lr

