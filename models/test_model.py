from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import numpy as np
import torch
import pdb
class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        if opt.which_model_netG == 'encode_decode_vessel_2scale_share_res':
            self.input_512 = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
            self.input_256 = self.Tensor(opt.batchSize, opt.input_nc, np.int(opt.fineSize/2), np.int(opt.fineSize/2))
            self.input_norm = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        elif opt.which_model_netG == 'encode_decode_vessel_res' or opt.which_model_netG =='encode_decode_res':
            self.input = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
            self.input_norm = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        # self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        elif opt.which_model_netG == 'encode_decode_vessel_2scale_share_res_atten':
            self.input_512 = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
            self.input_256 = self.Tensor(opt.batchSize, opt.input_nc, np.int(opt.fineSize/2), np.int(opt.fineSize/2))
            self.input_norm = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        elif opt.which_model_netG == 'encode_decode_vessel_2scale_share_res_atten_sum':
            self.input_512 = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
            self.input_256 = self.Tensor(opt.batchSize, opt.input_nc, np.int(opt.fineSize/2), np.int(opt.fineSize/2))
            self.input_norm = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.segmroot, opt.norm, not opt.no_dropout, self.gpu_ids,
                                      False, opt.learn_residual)
        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'G', which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        if self.opt.which_model_netG =='encode_decode_vessel_2scale_share_res' or self.opt.which_model_netG =='encode_decode_vessel_2scale_share_res_atten'\
                or self.opt.which_model_netG =='encode_decode_vessel_2scale_share_res_atten_sum':
            input_512 = input['A']
            temp = self.input_512.clone()
            temp.resize_(input_512.size()).copy_(input_512)
            self.input_512 = temp

            input_256 = input['A_256']
            temp_256 = self.input_256.clone()
            temp_256.resize_(input_256.size()).copy_(input_256)
            self.input_256 = temp_256

            input_norm = input['A']
            temp_ves = self.input_norm.clone()
            temp_ves.resize_(input_norm.size()).copy_(input_norm)
            temp_ves = temp_ves * 3.2 - 1.6
            self.input_norm = temp_ves
            self.image_paths = input['A_paths']
        elif self.opt.which_model_netG == 'encode_decode_vessel_res' or self.opt.which_model_netG =='encode_decode_res':
            input_512 = input['A']
            temp = self.input.clone()
            temp.resize_(input_512.size()).copy_(input_512)
            self.input = temp

            input_norm = input['A']
            temp_ves = self.input_norm.clone()
            temp_ves.resize_(input_norm.size()).copy_(input_norm)
            temp_ves = temp_ves * 3.2 - 1.6
            self.input_norm = temp_ves
            self.image_paths = input['A_paths']

    def test(self):
        if self.opt.which_model_netG =='encode_decode_vessel_2scale_share_res':
            self.real_A = Variable(self.input_512, requires_grad=False)
            self.real_A_256 = Variable(self.input_256, requires_grad=False)
            self.real_A_ves =Variable(self.input_norm,requires_grad=False)
            self.fake_B,self.fake_B_256, self.vess = self.netG.forward(self.real_A, self.real_A_256, self.real_A_ves)
        elif self.opt.which_model_netG == 'encode_decode_vessel_res':
            self.real_A = Variable(self.input, volatile=True)
            self.real_A_ves =Variable(self.input_norm,volatile=True)
            self.fake_B,self.vess = self.netG.forward(self.real_A, self.real_A_ves)
        elif self.opt.which_model_netG =='encode_decode_vessel_2scale_share_res_atten' or self.opt.which_model_netG =='encode_decode_vessel_2scale_share_res_atten_sum':
            self.real_A = Variable(self.input_512, requires_grad=False)
            self.real_A_256 = Variable(self.input_256, requires_grad=False)
            self.real_A_ves =Variable(self.input_norm,requires_grad=False)
            self.fake_B,self.fake_B_256, self.vess, self.spot = self.netG.forward(self.real_A, self.real_A_256, self.real_A_ves)
        elif self.opt.which_model_netG == 'encode_decode_res':
            self.real_A = Variable(self.input, volatile=True)
            self.real_A_ves =Variable(self.input_norm,volatile=True)
            self.fake_B= self.netG.forward(self.real_A)
    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        if self.opt.which_model_netG == 'encode_decode_vessel_2scale_share_res' :
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            vessel = util.tensor2im(self.vess.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('vessel', vessel)])
        elif self.opt.which_model_netG == 'encode_decode_vessel_res':
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            vessel = util.tensor2im(self.vess.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('vessel', vessel)])
        elif self.opt.which_model_netG == 'encode_decode_vessel_2scale_share_res_atten' or self.opt.which_model_netG =='encode_decode_vessel_2scale_share_res_atten_sum' :
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            vessel = util.tensor2im(self.vess.data)
            spot = util.tensor2im(self.spot.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('vessel', vessel), ('spot', spot)])
        elif self.opt.which_model_netG == 'encode_decode_res':
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])
            # return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('vessel', vessel)])
