import argparse
import os
from util import util
import torch

class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		self.initialized = False

	def initialize(self):
		#  test on degraded image folder：　color|blur|uneven|spot|fuse|
		# testing on drive dataset
		# self.parser.add_argument('--dataroot', type=str,default='/media/szy/project-data/medical-image-data/REFUGE/test/degrad3/',help='path to degraded imgs ON SERVER')
		# self.parser.add_argument('--maskroot', type=str,default='/media/szy/project-data/medical-image-data/REFUGE/test/mask/',help='path to degraded imgs ON SERVER')
		#
		# self.parser.add_argument('--dataroot', type=str,default='/home/szy/dataset/medical/tmi_test/drive/fuse2/',help='path to degraded imgs ON SERVER')
		# self.parser.add_argument('--maskroot', type=str,default='/home/szy/dataset/medical/tmi_test/drive/mask/',help='path to degraded imgs ON SERVER')
		# testing on kaggle dataset
		# self.parser.add_argument('--dataroot', type=str,default='/home/szy/dataset/medical/tmi_test/kaggle/fuse/',help='path to degraded imgs ON SERVER')
		# self.parser.add_argument('--maskroot', type=str,default='/home/szy/dataset/medical/tmi_test/kaggle/mask/',help='path to degraded imgs ON SERVER')
		# testing on real images

		# ori
		# self.parser.add_argument('--dataroot', type=str,default='/home/szy/dataset/medical/tmi_test/real/fuse/',help='path to degraded imgs ON SERVER')
		# self.parser.add_argument('--maskroot', type=str,default='/home/szy/dataset/medical/tmi_test/real/mask/',help='path to degraded imgs ON SERVER')

		self.parser.add_argument('--dataroot', type=str,default='/media/szy/zy_data/compare_real/0-3/',help='path to degraded imgs ON SERVER')
		self.parser.add_argument('--maskroot', type=str,default='/home/szy/dataset/medical/Kaggle/original_mask/all_mask/',help='path to degraded imgs ON SERVER')

		#  train
		# self.parser.add_argument('--dataroot', type=str,default='/home/szy/dataset/medical/Kaggle/dataset-transfer/train/',help='path to degraded imgs ON SERVER')
		self.parser.add_argument('--segmroot', type=str, default='/home/sz1/medical/code/weights/CE-Net-Vessel-DRIVE-85.th', help='path to degraded imgs ON MY WORK STATION')

		self.parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
		self.parser.add_argument('--loadSizeX', type=int, default=512, help='scale images to this size')
		self.parser.add_argument('--loadSizeY', type=int, default=512, help='scale images to this size')
		self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
		self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
		self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
		self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
		self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
		self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
		self.parser.add_argument('--which_model_netG', type=str, default='encode_decode_vessel_2scale_share_res_atten_sum', help='selects model to use for netG')
		# encode_decode_vessel_2scale_share_res_atten||encode_decode_vessel_2scale_share_res||encode_decode_vessel_share
		self.parser.add_argument('--checkpoints_dir', type=str,default='./encode_decode_vessel_2scale_atten10_dagrad3', help='models are saved here encode_decode_vessel_res')
		# encode_decode_vessel_addrelu_2scale_share_res2 ||encode_decode_vessel_addrelu_2scale_share_atten_res2

		self.parser.add_argument('--learn_residual', action='store_true', help='if specified, model would learn only the residual to the input')
		self.parser.add_argument('--gan_type', type=str, default='gan', help='wgan-gp : Wasserstein GAN with Gradient Penalty, lsgan : Least Sqaures GAN, gan : Vanilla GAN')
		self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
		self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
		self.parser.add_argument('--dataset_mode', type=str, default='single', help='chooses how datasets are loaded. [aliged_spera_single | aligned | single|aliged_spera_singlealiged_spera_single]')
		self.parser.add_argument('--model', type=str, default='test', help='chooses which model to use. pix2pix, test, content_gan,HDRCNN')
		self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
		self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')

		self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
		self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
		self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
		self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
		self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
		self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
		self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
		self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
		self.parser.add_argument('--resize_or_crop', type=str, default='scale', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|scale]')
		self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

		self.initialized = True

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()
		self.opt.isTrain = self.isTrain   # train or test

		str_ids = self.opt.gpu_ids.split(',')
		self.opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				self.opt.gpu_ids.append(id)

		# set gpu ids
		if len(self.opt.gpu_ids) > 0:
			torch.cuda.set_device(self.opt.gpu_ids[0])

		args = vars(self.opt)

		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')

		# save to the disk
		expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
		util.mkdirs(expr_dir)
		file_name = os.path.join(expr_dir, 'opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write('------------ Options -------------\n')
			for k, v in sorted(args.items()):
				opt_file.write('%s: %s\n' % (str(k), str(v)))
			opt_file.write('-------------- End ----------------\n')
		return self.opt
