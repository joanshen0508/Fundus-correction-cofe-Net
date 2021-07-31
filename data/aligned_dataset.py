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

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir_AB))

        #assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSizeX * 2, self.opt.loadSizeY), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)

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

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        # for testing dataloader
        # AB=util.tensor2im(AB)
        # util.save_image(AB,'/home/sz1/medical/code/whole.jpg')
        #
        # A=util.tensor2im(A)
        # util.save_image(A,'/home/sz1/medical/code/de.jpg')
        # B=util.tensor2im(B)
        # util.save_image(B,'/home/sz1/medical/code/GT.jpg')

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'

    def tensor_to_np(tensor):
        img = tensor.mul(255).byte()
        img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
        return img
