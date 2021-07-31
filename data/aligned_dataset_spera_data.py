import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import math
import matplotlib.pyplot as plt
import util.util as util

class AlignedDatasetspera(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        #assert(opt.resize_or_crop == 'resize_and_crop')

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

        # transform_list = [transforms.ToTensor(),
                          # transforms.Normalize((0.5, 0.5, 0.5),
                                               # (0.5, 0.5, 0.5))]

        # self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):

        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        B_path = self.B_paths[index % self.A_size]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)


        # AB_path = self.AB_paths[index]
        # AB = Image.open(AB_path).convert('RGB')
        # AB = AB.resize((self.opt.loadSizeX * 2, self.opt.loadSizeY), Image.BICUBIC)
        # AB = self.transform(AB)

        w_total = A_img.size(2)
        w = int(w_total / 2)
        h = A_img.size(1)

        thresh = min(h , w)
        h_thresh = int(h/2 - math.sqrt(thresh/2 * thresh/2- self.opt.fineSize/2 * self.opt.fineSize/2))
        h_offset = random.randint(h_thresh, thresh - h_thresh -self.opt.fineSize-1)

        h_delta = int(math.sqrt(thresh*thresh/4 - (h_offset-thresh/2)*(h_offset-thresh/2)))
        min_w= (thresh/2-h_delta)
        max_w = (thresh/2+h_delta-self.opt.fineSize-1)

        # print(min_w)
        # print(max_w)
        w_offset = random.randint(min(0,min_w),max(0,max_w))

        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = A_img[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = B_img[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]

        # for testing dataloader
        # AB=util.tensor2im(AB)
        # util.save_image(AB,'/home/sz1/medical/code/whole.jpg')
        #
        # A=util.tensor2im(A)
        # util.save_image(A,'/home/sz1/medical/code/de.jpg')
        # B=util.tensor2im(B)
        # util.save_image(B,'/home/sz1/medical/code/GT.jpg')

        # if (not self.opt.no_flip) and random.random() < 0.5:
        #     idx = [i for i in range(A.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     A = A.index_select(2, idx)
        #     B = B.index_select(2, idx)

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'AlignedDatasetspera'

    def tensor_to_np(tensor):
        img = tensor.mul(255).byte()
        img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
        return img
