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
import pdb
try:
	xrange          # Python2
except NameError:
	xrange = range  # Python 3

class HDRCC_2_share(BaseModel):
	def name(self):
		return 'HDRCNN_2_shareModel'

	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		self.isTrain = opt.isTrain
		# define tensors
		self.input_A = self.Tensor(opt.batchSize, opt.input_nc,	opt.fineSize, opt.fineSize)
		self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
		self.input_A_ves = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
		self.input_B_ves = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize) # BINARY_CLASS problem =2
		self.input_region = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
		self.input_disk = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
		#  scale 2
		self.input_A_256 = self.Tensor(opt.batchSize, opt.input_nc,	np.int(opt.fineSize/2), np.int(opt.fineSize/2))
		self.input_B_256 = self.Tensor(opt.batchSize, opt.output_nc,np.int(opt.fineSize/2), np.int(opt.fineSize/2))
		self.input_region_256 = self.Tensor(opt.batchSize, 1, np.int(opt.fineSize/2), np.int(opt.fineSize/2))
		self.input_disk_256 = self.Tensor(opt.batchSize, 1,np.int(opt.fineSize/2), np.int(opt.fineSize/2))
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
			self.contentLoss, self.segmenLoss = init_loss(opt, self.Tensor)

		print('---------- Networks initialized -------------')
		networks.print_network(self.netG)
		print('-----------------------------------------------')

	def set_input(self, input):
		AtoB = self.opt.which_direction == 'AtoB'
		input_A = input['A' if AtoB else 'B']
		input_B = input['B' if AtoB else 'A']
		input_A_ves = input['A_ves' if AtoB else 'B_ves']
		input_B_ves = input['B_ves' if AtoB else 'A_ves']
		input_region = input['B_region' if AtoB else 'A_region']
		input_disk = input['B_disk' if AtoB else 'A_disk']
		# SCALE 2
		input_A_256 =input['A_256' if AtoB else 'B_256']
		input_B_256 = input['B_256' if AtoB else 'A_256']
		input_region_256 = input['B_region_256' if AtoB else 'A_region_256']
		input_disk_256 = input['B_disk_256' if AtoB else 'A_disk_256']

		self.input_A.resize_(input_A.size()).copy_(input_A)
		self.input_B.resize_(input_B.size()).copy_(input_B)
		self.input_A_ves.resize_(input_A_ves.size()).copy_(input_A_ves)
		self.input_B_ves.resize_(input_B_ves.size()).copy_(input_B_ves)
		self.input_region.resize_(input_region.size()).copy_(input_region)
		self.input_disk.resize_(input_disk.size()).copy_(input_disk)
		# SCALE 2
		self.input_A_256.resize_(input_A_256.size()).copy_(input_A_256)
		self.input_B_256.resize_(input_B_256.size()).copy_(input_B_256)
		self.input_region_256.resize_(input_region_256.size()).copy_(input_region_256)
		self.input_disk_256.resize_(input_disk_256.size()).copy_(input_disk_256)

		#self.image_paths = input['A_paths' if AtoB else 'B_paths']

	def forward(self):
		self.real_A = Variable(self.input_A)
		self.real_A_ves =Variable(self.input_A_ves)
		# self.fake_B, self.fake_B_ves = self.netG.forward(self.real_A, self.real_A_ves)
		self.real_B = Variable(self.input_B)
		self.real_B_ves = Variable(self.input_B_ves)
		self.input_region = Variable(self.input_region)
		self.input_disk= Variable(self.input_disk)
	# 	SCALE 2
		self.real_A_256 =Variable(self.input_A_256)
		self.real_B_256 =Variable(self.input_B_256)
		self.input_region_256 =Variable(self.input_region_256)
		self.input_disk_256 = Variable(self.input_disk_256)
		self.fake_B,self.fake_B_256,self.fake_B_ves = self.netG.forward(self.real_A,self.real_A_256,self.real_A_ves)



	# no backprop gradients
	def test(self):
		self.real_A = Variable(self.input_A, volatile=True)
		self.real_A_256 =Variable(self.input_A_256, volatile=True)
		self.real_A_ves = Variable(self.input_A_ves, volatile=True)
		pdb.set_trace()
		self.fake_B,self.fake_B_256, self.fake_B_ves = self.netG.forward(self.real_A, self.real_A_256,self.real_A_ves)
		self.real_B = Variable(self.input_B, volatile=True)
		self.real_B_256 =Variable(self.input_B_256, volatile=True)
		self.real_B_ves = Variable(self.input_B_ves, volatile = True)

	# get image paths
	def get_image_paths(self):
		return self.image_paths

	def backward_G(self):
		# self.loss_G_GAN = self.discLoss.get_g_loss(self.netD, self.real_A, self.fake_B)
		# Second, G(A) = B
		# self.loss_G_Content = self.contentLoss.get_loss(self.fake_B, self.real_B) * self.opt.lambda_A
		self.loss_G_Content = self.contentLoss.get_loss(self.real_B* self.input_region, self.fake_B* self.input_region) #
		self.loss_G_Segmen = self.segmenLoss.get_loss( self.real_B_ves* self.input_region,self.fake_B_ves* self.input_region) * self.opt.lambda_S# * self.input_region
		self.loss_G_Disk_Content = self.contentLoss.get_loss(self.real_B* self.input_disk , self.fake_B* self.input_disk ) *10

		# sacle 2
		self.loss_G_Content_256 = self.contentLoss.get_loss(self.real_B_256* self.input_region_256, self.fake_B_256* self.input_region_256) #
		self.loss_G_Disk_Content_256 = self.contentLoss.get_loss(self.real_B_256* self.input_disk_256 , self.fake_B_256* self.input_disk_256 ) *10
		#self.loss_G_Percep = self.percepLoss.get_loss(self.fake_B, self.real_B)
		self.loss_G = self.loss_G_Content  + self.loss_G_Disk_Content+ self.loss_G_Segmen + self.loss_G_Content_256 + self.loss_G_Disk_Content_256

		self.loss_G.backward()

	def optimize_parameters(self):
		self.forward()
		self.optimizer_G.zero_grad()
		self.backward_G()
		self.optimizer_G.step()

	def get_current_errors(self):

		return OrderedDict([('G_total', self.loss_G.item()),
							('G_L1', self.loss_G_Content.item()),
							('G_L1_disk', self.loss_G_Disk_Content.item()),
							('G_L1_256', self.loss_G_Content_256.item()),
							('G_L1_disk_256', self.loss_G_Disk_Content_256.item()),
							('G_segmen', self.loss_G_Segmen.item()),
							])

	def get_current_visuals(self):
		real_A = util.tensor2im(self.real_A.data)
		# real_A_ves = util.tensor2im(self.real_A_ves.data)
		
		fake_B = util.tensor2im(self.fake_B.data)
		# fake_B_ves = util.tensor2im(self.fake_B_ves)
		
		real_B = util.tensor2im(self.real_B.data)		
		# real_B_ves = util.tensor2im(self.real_B_ves.data)

		fake_B_256 = util.tensor2im(self.fake_B_256.data)
		# fake_B_ves = util.tensor2im(self.fake_B_ves)

		real_B_256 = util.tensor2im(self.real_B_256.data)
		# real_B_ves = util.tensor2im(self.real_B_ves.data)
		return OrderedDict([('Blurred_Train', real_A), ('Restored_Train', fake_B), ('Sharp_Train', real_B),('Restored_Train_256', fake_B_256), ('Sharp_Train_256', real_B_256),('Restored_vessel', real_A),('sharp_vessel', real_A)])

	def save(self, label):
		self.save_network(self.netG, 'G', label, self.gpu_ids)

	def update_learning_rate(self):
		lrd = self.opt.lr / self.opt.niter_decay
		lr = self.old_lr - lrd
		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr
		print('update learning rate: %f -> %f' % (self.old_lr, lr))
		self.old_lr = lr

