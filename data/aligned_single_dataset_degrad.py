import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset,get_transform
from data.image_folder import make_dataset
from PIL import Image
import math
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import util.util as util
from motion_blur.degrad_image import DegradImage
from PIL import Image
import numpy as np
import cv2
import json

class AlignedDatasetspera_Single(BaseDataset):
    def initialize(self, opt):
        """ path infos:
        dir_B: gt retinal images
        dir_mask: vessel gt of retinal images
        dir_region: the valid mask (region) of retinal images
        dir_disk: disc coordinate for calculating additional local structual loss
        """
        self.opt = opt
        self.root = opt.dataroot
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_mask = os.path.join(opt.dataroot, opt.phase + 'A_ves_dis_com_512')
        self.dir_region = os.path.join(opt.dataroot, opt.phase + 'A_mask')
        self.dir_disk = os.path.join(opt.dataroot, opt.phase+'A_disk')

        self.B_paths = make_dataset(self.dir_B)
        self.mask_paths = make_dataset(self.dir_mask)
        self.region_paths = make_dataset(self.dir_region)
        self.disk_paths = make_dataset(self.dir_disk)
        #assert(opt.resize_or_crop == 'resize_and_crop')

        self.B_paths = sorted(self.B_paths)
        self.mask_paths = sorted(self.mask_paths)
        self.region_paths = sorted(self.region_paths)
        self.disk_paths = sorted(self.disk_paths)
        # self.B_paths = sorted(self.B_paths)

        self.B_size = len(self.B_paths)
        self.A_size = len(self.B_paths)
        self.mask_size = len(self.mask_paths)
        self.transform = get_transform(opt)
        # self.degrad = DegradImage(opt)

        # transform_list = [transforms.ToTensor(),
                          # transforms.Normalize((0.5, 0.5, 0.5),
                                               # (0.5, 0.5, 0.5))]

        # self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):

        B_path = self.B_paths[index % self.B_size]
        mask_path = self.mask_paths[index % self.B_size]
        region_path =self.region_paths[index % self.B_size]
        disk_path = self.disk_paths[index % self.B_size]
        index_B = index % self.B_size
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        B_img = Image.open(B_path).convert('RGB')
        (h,w) =B_img.size
        B_img = B_img.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)

        #### mask of the retina image
        B_region = Image.open(region_path).convert('L')
        B_region = B_region.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
        B_region = np.expand_dims(B_region, axis=2)
        B_region_256 = cv2.resize(B_region,(np.int(self.opt.fineSize/2), np.int(self.opt.fineSize/2)))
        B_region_256 = np.expand_dims(B_region_256, axis=2)
        B_region = np.array(B_region, np.float32).transpose(2, 0, 1) / 255.0
        B_region_256 = np.array(B_region_256, np.float32).transpose(2, 0, 1) / 255.0
        ### mask of the disk
        with open(disk_path) as f:
            coor = json.load(f)
        coor[0] = math.floor(coor[0] * self.opt.loadSizeY / h)
        coor[1] = math.floor(coor[1] * self.opt.loadSizeY / h)
        coor[2] = math.floor(coor[2] * self.opt.loadSizeX / w)
        coor[3] = math.floor(coor[3] * self.opt.loadSizeX / w)
        B_disk = np.zeros((self.opt.loadSizeY, self.opt.loadSizeX))
        B_disk[coor[0]:coor[1],coor[2]:coor[3]]=1
        B_disk = np.expand_dims(B_disk, axis=2)
        B_disk_256 = cv2.resize(B_disk, (np.int(self.opt.fineSize / 2), np.int(self.opt.fineSize / 2)))
        B_disk = np.array(B_disk, np.float32).transpose(2, 0, 1)
        B_disk_256 = np.expand_dims(B_disk_256, axis=2)
        B_disk_256 = np.array(B_disk_256, np.float32).transpose(2, 0, 1)

        #### mask of the vessel for segmentation
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = cv2.resize(mask,(self.opt.fineSize, self.opt.fineSize))
        mask = np.expand_dims(mask, axis=2)
        mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
        #
        mask[mask >= 0.6] = 1
        mask[mask < 0.4] = 0

        A_img =DE_COLOR(B_img,0.5,0.5,0.5)
        A_img = self.transform(A_img)
        B_img = self.transform(B_img)
        A_img = A_img.numpy()
        B_img = B_img.numpy()

        A_img = DE_SPOT(A_img, self.opt.loadSizeY, self.opt.loadSizeX)
        A_img = DE_HALO(A_img, self.opt.loadSizeY, self.opt.loadSizeX)
        A_img = DE_BLUR(A_img, self.opt.loadSizeY, self.opt.loadSizeX)
        A_img = DE_HOLE(A_img, self.opt.loadSizeY, self.opt.loadSizeX)

        # scale 2
        A_img_256 = np.array(A_img).transpose(1, 2, 0)
        A_img_256 = cv2.resize(A_img_256,(np.int(self.opt.fineSize/2), np.int(self.opt.fineSize/2)))
        A_img_256 = np.array(A_img_256).transpose(2,0,1)

        B_img_256 = np.array(B_img).transpose(1, 2, 0)
        B_img_256 = cv2.resize(B_img_256,(np.int(self.opt.fineSize/2), np.int(self.opt.fineSize/2)))
        B_img_256 = np.array(B_img_256).transpose(2,0,1)



        A_ves_img = A_img * 3.2 - 1.6

        # w_total = np.size(B_img,2)
        # w = int(w_total)
        # h = np.size(B_img,1)

        # thresh = min(h , w)
        # h_thresh = int(h/2 - math.sqrt(thresh/2 * thresh/2- self.opt.fineSize/2 * self.opt.fineSize/2))
        # h_offset = random.randint(h_thresh, thresh - h_thresh -self.opt.fineSize-1)
        #
        # h_delta = int(math.sqrt(thresh*thresh/4 - (h_offset-thresh/2)*(h_offset-thresh/2)))
        # min_w= (thresh/2-h_delta)
        # max_w = (thresh/2+h_delta-self.opt.fineSize-1)

        # w_offset = random.randint(min(0,min_w),max(0,max_w))
        # w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        # h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        # A_img = (np.transpose(A_img, (1, 2, 0)))
        # B_img = (np.transpose(B_img, (1, 2, 0)))
        # A = A_img[ h_offset:h_offset + self.opt.fineSize,
        #        w_offset:w_offset + self.opt.fineSize,:]
        # B = B_img[ h_offset:h_offset + self.opt.fineSize,
        #        w_offset:w_offset + self.opt.fineSize,:]



        # A = A_img[ h_offset:h_offset + self.opt.fineSize,
        #        w_offset:w_offset + self.opt.fineSize,:]
        # B = B_img[ h_offset:h_offset + self.opt.fineSize,
        #        w_offset:w_offset + self.opt.fineSize,:]

        # SAVE FOR TEST beign
        # A_img=util.numpy2im(A_img)
        # B_img=util.numpy2im(B_img)
        # A_ves =util.numpy2im((A_ves_img+1.6)/3.2)
        # B_ves = util.numpy2im(mask)
        # B_region = util.numpy2im(B_region)
        # B_disk = util.numpy2im(B_disk)
        # A_img_256=util.numpy2im(A_img_256)
        # B_img_256=util.numpy2im(B_img_256)
        # B_region_256 = util.numpy2im(B_region_256)
        # B_disk_256 = util.numpy2im(B_disk_256)

        # util.save_image(A,'/home/sz1/medical/code/DE-patch.jpg')
        # util.save_image(B,'/home/sz1/medical/code/ORI-patch.jpg')
        # util.save_image(A_img, './A_degraded.jpg')
        # util.save_image(B_img, './B.jpg')
        # util.save_image(A_ves, './vessel_input.jpg')
        # util.save_image(B_ves, './vessel_GT.jpg')
        # util.save_image(B_region,'./B_mask.jpg')
        # util.save_image(B_disk,'./B_disk.jpg')
        # util.save_image(A_img_256, './A_degraded_256.jpg')
        # util.save_image(B_img_256, './B_256.jpg')
        # util.save_image(B_region_256,'./B_mask_256.jpg')
        # util.save_image(B_disk_256,'./B_disk_256.jpg')
        # SAVE FOR TEST END

        # if (not self.opt.no_flip) and random.random() < 0.5:
        #     idx = [i for i in range(A.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     A = A.index_select(2, idx)
        #     B = B.index_select(2, idx)

        # A_img:degraded image
        # B_img: GT image
        # A_ves_img: vessel
        # B_region: mask of retinal images
        # B_disk:retinal_dis region
        return {'A': A_img, 'B': B_img,'A_256':A_img_256, 'B_256':B_img_256,'B_paths':B_path, 'A_ves': A_ves_img, 'B_ves':mask,'B_region':B_region,'B_disk':B_disk,'B_disk_256':B_disk_256,'B_region_256':B_region_256}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'AlignedDatasetspera_Single'

    def tensor_to_np(tensor):
        img = tensor.mul(255).byte()
        img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
        return img

def DE_COLOR(img, brightness=0.0, contrast=0.0, saturation=0.0):
    """Randomly change the brightness, contrast and saturation of an image"""
    if brightness > 0:
        brightness_factor = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
        img = F.adjust_brightness(img, brightness_factor)
    if contrast > 0:
        contrast_factor = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
        img = F.adjust_contrast(img, contrast_factor)
    if saturation > 0:
        saturation_factor = random.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation)
        img = F.adjust_saturation(img, saturation_factor)
    return img

def DE_SPOT(img, h, w, center=None, radius=None):
    s_num =random.randint(5,20)
    for i in range(s_num):
        # if radius is None: # use the smallest distance between the center and image walls
            # radius = min(center[0], center[1], w-center[0], h-center[1])
        radius = random.randint(math.ceil(0.026*h),int(0.05*h))

        # if center is None: # use the middle of the image
        center  = [random.randint(radius+1,w-radius-1),random.randint(radius+1,h-radius-1)]
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        circle = dist_from_center <= (int(radius/2))

        k = 0.7 + (1.0 - 0.7) * random.random()
        beta = 0.5 + (1.5 - 0.5) * random.random()
        A = k *np.ones((3,1))
        d = 0.3 * random.random()
        t = math.exp(-beta * d)

        mask = np.zeros((h, w))
        mask[circle] = np.multiply(A[0],(1-t))
        mask = cv2.GaussianBlur(mask, (5, 5), 1.8)
        mask = np.array([mask,mask,mask])
        img = img + mask
        img = np.maximum(img,0)
        img = np.minimum(img,1)
    return img

def DE_HALO(img, h, w, center=None, radius=None):
    w0_a = random.randint(w/2-int(w/8),w/2+int(w/8))
    h0_a = random.randint(h/2-int(h/8),h/2+int(h/8))
    center_a = [w0_a, h0_a]

    wei_dia_a =0.75 + (1.0-0.75) * random.random()
    dia_a = min(h,w)*wei_dia_a
    Y_a, X_a = np.ogrid[:h, :w]
    dist_from_center_a = np.sqrt((X_a - center_a[0]) ** 2 + (Y_a - center_a[1]) ** 2)
    circle_a = dist_from_center_a <= (int(dia_a / 2))

    mask_a = np.zeros((h, w))
    mask_a[circle_a] = np.mean(img) #np.multiply(A[0], (1 - t))

    center_b =center_a
    Y_b, X_b = np.ogrid[:h, :w]
    dist_from_center_b = np.sqrt((X_b - center_b[0]) ** 2 + (Y_b - center_b[1]) ** 2)

    dia_b_max =2* int(np.sqrt(max(center_a[0],h-center_a[0])*max(center_a[0],h-center_a[0])+max(center_a[1],h-center_a[1])*max(center_a[1],w-center_a[1])))/min(w,h)
    wei_dia_b = 1.0+(dia_b_max-1.0) * random.random()
    dia_b =min(h,w)* wei_dia_b +abs(max(center_b[0]-w/2,center_b[1]-h/2)+max(w,h)/2)

    circle_b = dist_from_center_b <= (int(dia_b / 2))

    mask_b = np.zeros((h, w))
    mask_b[circle_b] = np.mean(img)#np.multiply(A[0], (1 - t))
    delta_circle =np.abs(mask_a-mask_b)

    # 3_sigma_min =dia_energy/2 (or, rad_energy) 3_sigma_max = dia_energy
    sigma = random.randint(int(min(w,h)/6),min(w,h)/2)/3
    gauss_rad = int(sigma * 1.5)
    if(gauss_rad % 2) == 0:
        gauss_rad= gauss_rad+1
    delta_circle = cv2.GaussianBlur(delta_circle, (gauss_rad,gauss_rad), sigma)

    weight_r = [255/255,141/255,162/255]
    weight_g = [255/255,238/255,205/255]
    weight_b = [255/255,238/255,90/255]

    num = random.randint(0,2)
    delta_circle = np.array([weight_r[num]*delta_circle,weight_g[num]*delta_circle,weight_b[num]*delta_circle])
    img = img + delta_circle

    img = np.maximum(img, 0)
    img = np.minimum(img, 1)

    return img

def DE_BLUR(img, h, w, center=None, radius=None):
    img = (np.transpose(img, (1, 2, 0)))

    sigma = 0+(15-0) * random.random()
    rad_w = random.randint(int(sigma/3), int(sigma/2))
    rad_h = random.randint(int(sigma/3), int(sigma/2))
    if (rad_w % 2) == 0: rad_w = rad_w + 1
    if(rad_h % 2) ==0 : rad_h =rad_h + 1

    img = cv2.GaussianBlur(img, (rad_w,rad_h), sigma)
    img = (np.transpose(img, (2, 0, 1)))

    img = np.maximum(img, 0)
    img= np.minimum(img, 1)

    return img

def DE_HOLE(img, h, w, center=None, diameter=None):
    # if radius is None: # use the smallest distance between the center and image walls
    diameter_circle = random.randint(int(0.3*w), int(0.5 * w))
    center =[random.randint(1,w-1),random.randint(1,h-1)]
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    circle = dist_from_center <= (int(diameter_circle/2))

    mask = np.zeros((h, w))
    mask[circle] = 1

    # scipy.misc.imsave('/home/szy/code/medical/pytorch-CycleGAN-and-pix2pix/mask_1.jpg',mask)
    brightness = -0.05
    brightness_factor = random.uniform((brightness-0.2*1), min(brightness, 0))

    mask = mask * brightness_factor
    # scipy.misc.imsave('/home/szy/code/medical/pytorch-CycleGAN-and-pix2pix/mask_2.jpg', mask)
    sigma = random.uniform(diameter_circle/4, diameter_circle/3)

    rad_w = random.randint(int(diameter_circle/4), int(diameter_circle/3))
    rad_h = random.randint(int(diameter_circle/4), int(diameter_circle/3))
    if (rad_w % 2) == 0: rad_w = rad_w + 1
    if(rad_h % 2) ==0 : rad_h =rad_h + 1

    mask = cv2.GaussianBlur(mask, (rad_w, rad_h), sigma)
    # scipy.misc.imsave('./mask_3.jpg',mask)
    mask = np.array([mask, mask, mask])
    img = img + mask
    img = np.maximum(img, 0)
    img = np.minimum(img, 1)

    return img
