import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.autograd as autograd
import numpy as np
import torchvision.models as models
import util.util as util
from util.image_pool import ImagePool
from torch.autograd import Variable
###############################################################################
# Functions
###############################################################################

class ContentLoss():
	def initialize(self, loss):
		self.criterion = loss
			
	def get_loss(self, fakeIm, realIm):
		return self.criterion(fakeIm, realIm)

class PerceptualLoss():
	
	def contentFunc(self):
		conv_3_3_layer = 14
		cnn = models.vgg19(pretrained=True).features
		cnn = cnn.cuda()
		model = nn.Sequential()
		model = model.cuda()
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
			if i == conv_3_3_layer:
				break
		return model
		
	def initialize(self, loss):
		self.criterion = loss
		self.contentFunc = self.contentFunc()
			
	def get_loss(self, fakeIm, realIm):
		f_fake = self.contentFunc.forward(fakeIm)
		f_real = self.contentFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return loss
		
class GANLoss(nn.Module):
	def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0,
				 tensor=torch.FloatTensor):
		super(GANLoss, self).__init__()
		self.real_label = target_real_label
		self.fake_label = target_fake_label
		self.real_label_var = None
		self.fake_label_var = None
		self.Tensor = tensor
		if use_l1:
			self.loss = nn.L1Loss()
		else:
			self.loss = nn.BCELoss()

	def get_target_tensor(self, input, target_is_real):
		target_tensor = None
		if target_is_real:
			create_label = ((self.real_label_var is None) or
							(self.real_label_var.numel() != input.numel()))
			if create_label:
				real_tensor = self.Tensor(input.size()).fill_(self.real_label)
				self.real_label_var = Variable(real_tensor, requires_grad=False)
			target_tensor = self.real_label_var
		else:
			create_label = ((self.fake_label_var is None) or
							(self.fake_label_var.numel() != input.numel()))
			if create_label:
				fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
				self.fake_label_var = Variable(fake_tensor, requires_grad=False)
			target_tensor = self.fake_label_var
		return target_tensor

	def __call__(self, input, target_is_real):
		target_tensor = self.get_target_tensor(input, target_is_real)
		return self.loss(input, target_tensor)

class DiscLoss():
	def name(self):
		return 'DiscLoss'

	def initialize(self, opt, tensor):
		self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)
		self.fake_AB_pool = ImagePool(opt.pool_size)
		
	def get_g_loss(self,net, realA, fakeB):
		# First, G(A) should fake the discriminator
		pred_fake = net.forward(fakeB)
		return self.criterionGAN(pred_fake, 1)
		
	def get_loss(self, net, realA, fakeB, realB):
		# Fake
		# stop backprop to the generator by detaching fake_B
		# Generated Image Disc Output should be close to zero
		self.pred_fake = net.forward(fakeB.detach())
		self.loss_D_fake = self.criterionGAN(self.pred_fake, 0)

		# Real
		self.pred_real = net.forward(realB)
		self.loss_D_real = self.criterionGAN(self.pred_real, 1)

		# Combined loss
		self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
		return self.loss_D
		
class DiscLossLS(DiscLoss):
	def name(self):
		return 'DiscLossLS'

	def initialize(self, opt, tensor):
		DiscLoss.initialize(self, opt, tensor)
		self.criterionGAN = GANLoss(use_l1=True, tensor=tensor)
		
	def get_g_loss(self,net, realA, fakeB):
		return DiscLoss.get_g_loss(self,net, realA, fakeB)
		
	def get_loss(self, net, realA, fakeB, realB):
		return DiscLoss.get_loss(self, net, realA, fakeB, realB)
		
class DiscLossWGANGP(DiscLossLS):
	def name(self):
		return 'DiscLossWGAN-GP'

	def initialize(self, opt, tensor):
		DiscLossLS.initialize(self, opt, tensor)
		self.LAMBDA = 10
		
	def get_g_loss(self, net, realA, fakeB):
		# First, G(A) should fake the discriminator
		self.D_fake = net.forward(fakeB)
		return -self.D_fake.mean()
		
	def calc_gradient_penalty(self, netD, real_data, fake_data):
		alpha = torch.rand(1, 1)
		alpha = alpha.expand(real_data.size())
		alpha = alpha.cuda()

		interpolates = alpha * real_data + ((1 - alpha) * fake_data)

		interpolates = interpolates.cuda()
		interpolates = Variable(interpolates, requires_grad=True)
		
		disc_interpolates = netD.forward(interpolates)

		gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
								  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
								  create_graph=True, retain_graph=True, only_inputs=True)[0]

		gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
		return gradient_penalty
		
	def get_loss(self, net, realA, fakeB, realB):
		self.D_fake = net.forward(fakeB.detach())
		self.D_fake = self.D_fake.mean()
		
		# Real
		self.D_real = net.forward(realB)
		self.D_real = self.D_real.mean()
		# Combined loss
		self.loss_D = self.D_fake - self.D_real
		gradient_penalty = self.calc_gradient_penalty(net, realB.data, fakeB.data)
		return self.loss_D + gradient_penalty

def init_loss(opt, tensor):
	disc_loss = None
	content_loss = None
	percep_loss = None
	
	if opt.model == 'pix2pix':
		content_loss = ContentLoss()
		content_loss.initialize(nn.L1Loss())
	elif opt.model =='HDRCNN' or opt.model =='HDRCNN_2SCALE_SHARE' :
		content_loss = ContentLoss()
		content_loss.initialize(nn.L1Loss())
		segmen_loss =dice_bce_loss()
		segmen_loss.initialize()
		return content_loss, segmen_loss
		#percep_loss =PerceptualLoss()
		#percep_loss.initialize(nn.MSELoss())
	elif opt.model =='HDRCNN_GEN':
		content_loss = ContentLoss()
		content_loss.initialize(nn.L1Loss())
		return content_loss


	else :
		raise ValueError("Model [%s] not recognized." % opt.model)

class dice_bce_loss(nn.Module):
    def initialize(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1)


    def multi_class_one_hot(self, label, classes):
        N, H, W = label.size(0), label.size(2), label.size(3)

        # y_hot = torch.LongTensor(N, classes, H, W).cuda()
        y_hot = torch.cuda.FloatTensor(N, classes, H, W)
        y_hot.zero_()
        # y_hot = y_hot.type(torch.cuda.LongTensor)
        label = label.type(torch.cuda.LongTensor)
        y_hot.scatter_(1, label.view(N, 1, H, W), 1)

        return y_hot

    def multi_class_dice_loss(self, input, mask):
        assert input.size() == mask.size(), "Input sizes must be equal to mask, the input size is {}, and the" \
                                            "mask size is {}".format(input.size(), mask.size())

        assert input.dim() == 4, "Input must be a 4D tensor"

        num = (input * mask).sum(dim=3).sum(dim=2)
        den1 = input.pow(2)
        den2 = mask.pow(2)

        dice = 2 * (num / (den1 + den2).sum(dim=3).sum(dim=2))
        return 1. - dice.sum() / (dice.size(1) * dice.size(0))
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

        
    def get_loss(self, y_true, y_pred):

        y_prediction = self.softmax(y_pred)
        # print("=====================")
        # print(y_true.size(),y_pred.size(),y_prediction.size())
        # print("=====================")
        y_mask_one_hot = self.multi_class_one_hot(y_true, classes=12)

        y_ce_true = y_true.squeeze(dim=1).float()

        # print(y_prediction.size())
        # print(y_mask_one_hot.size())

        a = self.bce_loss(y_pred, y_ce_true)
        # b = self.multi_class_dice_loss(y_prediction, y_mask_one_hot)
        return a


