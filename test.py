import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR
# from ssim import SSIM
from PIL import Image
from util import util
import numpy as np
import cv2

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
model = create_model(opt)
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
avgPSNR = 0.0
avgSSIM = 0.0
counter = 0

for i, data in enumerate(dataset):
	if i==28:
		break
	if i >= opt.how_many:
		break
	counter = i
	model.set_input(data)
	model.test()
	visuals = model.get_current_visuals()

	img_path = model.get_image_paths()
	folder, img_name =os.path.split(img_path[0])
	############## load mask for test########################
	# mask_dir = os.path.join(opt.maskroot, img_name[0:-11] + '_mask.png')
	mask_dir = os.path.join(opt.maskroot, img_name[0:-4] + '.png')
	mask = np.array(Image.open(mask_dir).convert('L'))
	mask = cv2.resize(mask, (opt.fineSize, opt.fineSize))
	mask[mask >= 255] = 255
	mask[mask != 255] = 0

	print('process image... %s' % img_path)
	B_img=util.numpy2im(visuals['fake_B'])
	A_img=util.numpy2im(visuals['real_A'])
	# A_vessel =util.numpy2im(visuals['vessel'])

	mask = np.expand_dims(mask, axis=2)
	mask = np.array(mask, np.float32)/255.0

	B_img = visuals['fake_B']*mask
	B_img =B_img.astype(np.uint8)

	A_img = visuals['real_A']*mask
	A_img =A_img.astype(np.uint8)

	# vessel = visuals['vessel']
	# vessel =(vessel,vessel,vessel)
	# vessel =np.array(vessel)
	# A_vessel =vessel[:,:,:,0]
	# A_vessel = (np.transpose(A_vessel, (1, 2, 0)))*mask
	# A_vessel =A_vessel.astype(np.uint8)

 	# atten result

	# mask_256 =cv2.resize(mask,(256,256))
	# mask_256 = np.expand_dims(mask_256,axis=2)
	# atten = visuals['spot']
	# atten =(atten,atten,atten)
	# atten =np.array(atten)
	# A_atten =atten[:,:,:,0]
	# A_atten = (np.transpose(A_atten, (1, 2, 0)))*mask_256
	# A_atten = A_atten.astype(np.uint8)


	# dir_A = os.path.join(opt.results_dir, '%s_ori%s' % (img_name[0:-5],'.png'))
	# util.save_image(A_img,dir_A)

	dir_B = os.path.join(opt.results_dir, '%s%s' % (img_name[0:-5], '_enhance.png'))
	util.save_image(B_img, dir_B)

	# dir_V = os.path.join(opt.results_dir, '%s_final_ves_sum%s' % (img_name[0:-5], '.png'))
	# util.save_image(A_vessel, dir_V)
	#
	# dir_S = os.path.join(opt.results_dir, '%s_final_att_sum%s' % (img_name[0:-5], '.png'))
	# util.save_image(A_atten, dir_S)

	#avgPSNR += PSNR(visuals['fake_B'],visuals['real_B'])
	#pilFake = Image.fromarray(visuals['fake_B'])
	#pilReal = Image.fromarray(visuals['real_B'])
	#avgSSIM += SSIM(pilFake).cw_ssim_value(pilReal)
	# visualizer.save_images(webpage, visuals, img_path)

	# img_path = model.get_image_paths()
	# dir_A = os.path.join(opt.results_dir, '%s%s' % (img_name[0:-5], '.png'))
	# print('process image... %s' % img_path)
	# util.save_image(A_img, dir_A)

#avgPSNR /= counter
#avgSSIM /= counter
#print('PSNR = %f, SSIM = %f' %
#				  (avgPSNR, avgSSIM))

webpage.save()
