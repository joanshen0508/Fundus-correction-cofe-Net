import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from torchvision import models
from functools import partial
import torch.nn.functional as F

###############################################################################
# Functions
###############################################################################
nonlinearity = partial(F.relu, inplace=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def load_pretrained_personal_model(model, path, use_gpu):
    if use_gpu:
        pretrain_dict = torch.load(path)
    else:
        pretrain_dict = torch.load(path, map_location=torch.device('cpu'))
    model_dict = {}
    state_dict = model.state_dict()
    for k, v in pretrain_dict.items():
        k_ori = k[7:]
        if k_ori in state_dict:
            model_dict[k_ori] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, segmdataroot, norm='batch', use_dropout=False, gpu_ids=[],
             use_parallel=True, learn_residual=False):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                               gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'resnet_15blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=15,
                               gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'encode_decode_vessel':
        netG = Encode_Decode_seg(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                                 gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'encode_decode_vessel_res':
        netG = Encode_Decode_seg_res(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                     n_blocks=9, gpu_ids=gpu_ids, use_parallel=use_parallel,
                                     learn_residual=learn_residual)
    elif which_model_netG == 'encode_decode':
        netG = Encode_Decode(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                             gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'encode_decode_res':
        netG = Encode_Decode_res(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                                 gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'encode_decode_vessel_2scale_share':
        netG = Encode_Decode_seg_2scale_share(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                              n_blocks=9, gpu_ids=gpu_ids, use_parallel=use_parallel,
                                              learn_residual=learn_residual)
    elif which_model_netG == 'encode_decode_vessel_2scale_share_res':
        netG = Encode_Decode_seg_2scale_share_res(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                                  use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids,
                                                  use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'encode_decode_vessel_2scale_share_res_atten':
        netG = Encode_Decode_seg_2scale_share_atten_res(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                                        use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids,
                                                        use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'encode_decode_vessel_2scale_share_res_atten_sum':
        netG = Encode_Decode_seg_2scale_share_atten_res_sum(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                                            use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids,
                                                            use_parallel=use_parallel, learn_residual=learn_residual)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)

    if which_model_netG == 'encode_decode_vessel':
        load_pretrained_personal_model(netG, segmdataroot, use_gpu)
    if which_model_netG == 'encode_decode_vessel_res':
        load_pretrained_personal_model(netG, segmdataroot, use_gpu)
    if which_model_netG == 'encode_decode_vessel_2scale_share':
        load_pretrained_personal_model(netG, segmdataroot, use_gpu)
    if which_model_netG == 'encode_decode_vessel_2scale_share_res':
        load_pretrained_personal_model(netG, segmdataroot, use_gpu)
    if which_model_netG == 'encode_decode_vessel_2scale_share_res_atten':
        load_pretrained_personal_model(netG, segmdataroot, use_gpu)
    if which_model_netG == 'encode_decode_vessel_2scale_share_res_atten_sum':
        load_pretrained_personal_model(netG, segmdataroot, use_gpu)
    return netG


# def define_D(input_nc, ndf, which_model_netD,
# 			 n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[], use_parallel = True):
# 	netD = None
# 	use_gpu = len(gpu_ids) > 0
# 	norm_layer = get_norm_layer(norm_type=norm)

# 	if use_gpu:
# 		assert(torch.cuda.is_available())
# 	if which_model_netD == 'basic':
# 		netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, use_parallel=use_parallel)
# 	elif which_model_netD == 'n_layers':
# 		netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, use_parallel=use_parallel)
# 	else:
# 		raise NotImplementedError('Discriminator model name [%s] is not recognized' %
# 								  which_model_netD)
# 	if use_gpu:
# 		netD.cuda(gpu_ids[0])
# 	netD.apply(weights_init)
# 	return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output, min=-1, max=1)
        return output


class Encode_Decode_seg(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Encode_Decode_seg, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.G_conv1 = nn.Conv2d(3, 32, 5, padding=2, bias=use_bias)
        self.G_relu1 = nn.ReLU()
        self.G_conv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv2_0 = nn.Conv2d(32, 64, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu2_0 = nn.ReLU()
        # concat 1/2
        self.G_conv2 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
        self.G_relu2 = nn.ReLU()
        self.G_conv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv3_0 = nn.Conv2d(64, 64, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu3_0 = nn.ReLU()
        # concat 1/4
        self.G_conv3 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
        self.G_relu3 = nn.ReLU()
        self.G_conv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv4_0 = nn.Conv2d(64, 128, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu4_0 = nn.ReLU()
        # concat 1/8
        self.G_conv4 = nn.Conv2d(256, 128, 5, padding=2, bias=use_bias)
        self.G_relu4 = nn.ReLU()
        self.G_conv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_deconv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_0 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=use_bias)

        self.G_deconv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_0 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=use_bias)

        self.G_deconv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_0 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=use_bias)

        self.G_deconv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_0 = nn.Conv2d(32, 3, 5, padding=2, bias=use_bias)

        #### CE_Net ######

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])

        self.decoder2 = DecoderBlock(filters[1], filters[0])

        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, input, input_norm):

        ### begin segmentation network ###
        x = self.firstconv(input_norm)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        out = F.sigmoid(out)

        ###　end segmentation network ###

        ### begin enhance network ###

        x = self.G_conv1(input)
        x = self.G_relu1(x)
        x = self.G_conv1_1(x)
        x = self.G_conv1_2(x)
        x = self.G_conv1_3(x)

        x = self.G_conv2_0(x)
        x = self.G_relu2_0(x)
        con_2 = torch.cat([x, d1], 1)
        x = self.G_conv2(con_2)
        x = self.G_relu2(x)
        x = self.G_conv2_1(x)
        x = self.G_conv2_2(x)
        x = self.G_conv2_３(x)

        x = self.G_conv3_0(x)
        x = self.G_relu３_0(x)
        con_4 = torch.cat([x, d2], 1)
        x = self.G_conv3(con_4)
        x = self.G_relu3(x)
        x = self.G_conv3_1(x)
        x = self.G_conv3_2(x)
        x = self.G_conv3_3(x)

        x = self.G_conv4_0(x)
        x = self.G_relu４_0(x)
        con_8 = torch.cat([x, d3], 1)
        x = self.G_conv4(con_8)
        x = self.G_relu4(x)
        x = self.G_conv4_1(x)
        x = self.G_conv4_2(x)
        x = self.G_conv4_3(x)

        x = self.G_deconv4_3(x)
        x = self.G_deconv4_2(x)
        x = self.G_deconv4_1(x)
        x = self.G_deconv4_0(x)

        x = self.G_deconv3_3(x)
        x = self.G_deconv3_2(x)
        x = self.G_deconv3_1(x)
        x = self.G_deconv3_0(x)

        x = self.G_deconv2_3(x)
        x = self.G_deconv2_2(x)
        x = self.G_deconv2_1(x)
        x = self.G_deconv2_0(x)

        x = self.G_deconv1_3(x)
        x = self.G_deconv1_2(x)
        x = self.G_deconv1_1(x)
        x = self.G_deconv1_0(x)
        output = F.sigmoid(x)
        ##＃end enhancement network ###

        ######copy channel to RGB
        # out_a =torch.unsqueeze(out[:,1,:,:],1)

        if self.learn_residual:
            output = input + output
            output = torch.clamp(output, min=-1, max=1)

        # output enhancement results
        # out segmentation result
        # fusion feature d3 d2 d1
        # d3 1/8 feature 128 channel
        # d2 1/4 feature 64 channel
        # d1 1/2 feature 64 channel
        return output, out


class Encode_Decode_seg_res(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Encode_Decode_seg_res, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.G_conv1 = nn.Conv2d(3, 32, 5, padding=2, bias=use_bias)
        self.G_relu1 = nn.ReLU()
        self.G_conv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv2_0 = nn.Conv2d(32, 64, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu2_0 = nn.ReLU()
        # concat 1/2
        self.G_conv2 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
        self.G_relu2 = nn.ReLU()
        self.G_conv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv3_0 = nn.Conv2d(64, 64, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu3_0 = nn.ReLU()
        # concat 1/4
        self.G_conv3 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
        self.G_relu3 = nn.ReLU()
        self.G_conv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv4_0 = nn.Conv2d(64, 128, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu4_0 = nn.ReLU()
        # concat 1/8
        self.G_conv4 = nn.Conv2d(256, 128, 5, padding=2, bias=use_bias)
        self.G_relu4 = nn.ReLU()
        self.G_conv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_deconv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_0 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=use_bias)

        self.G_deconv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_0 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=use_bias)

        self.G_deconv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_0 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=use_bias)

        self.G_deconv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_0 = nn.Conv2d(32, 3, 5, padding=2, bias=use_bias)

        #### CE_Net ######

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])

        self.decoder2 = DecoderBlock(filters[1], filters[0])

        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, input, input_norm):

        ### begin segmentation network ###
        x = self.firstconv(input_norm)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        out = F.sigmoid(out)

        ###　end segmentation network ###

        ### begin enhance network ###

        x = self.G_conv1(input)
        x = self.G_relu1(x)
        x = self.G_conv1_1(x)
        x = self.G_conv1_2(x)
        x_512 = self.G_conv1_3(x)

        x = self.G_conv2_0(x_512)
        x = self.G_relu2_0(x)
        con_2 = torch.cat([x, d1], 1)
        x = self.G_conv2(con_2)
        x = self.G_relu2(x)
        x = self.G_conv2_1(x)
        x = self.G_conv2_2(x)
        x_256 = self.G_conv2_３(x)

        x = self.G_conv3_0(x_256)
        x = self.G_relu3_0(x)
        con_4 = torch.cat([x, d2], 1)
        x = self.G_conv3(con_4)
        x = self.G_relu3(x)
        x = self.G_conv3_1(x)
        x = self.G_conv3_2(x)
        x_128 = self.G_conv3_3(x)

        x = self.G_conv4_0(x_128)
        x = self.G_relu4_0(x)
        con_8 = torch.cat([x, d3], 1)
        x = self.G_conv4(con_8)
        x = self.G_relu4(x)
        x = self.G_conv4_1(x)
        x = self.G_conv4_2(x)
        x = self.G_conv4_3(x)

        x = self.G_deconv4_3(x)
        x = self.G_deconv4_2(x)
        x = self.G_deconv4_1(x)
        x = self.G_deconv4_0(x)

        x = x + x_128

        x = self.G_deconv3_3(x)
        x = self.G_deconv3_2(x)
        x = self.G_deconv3_1(x)
        x = self.G_deconv3_0(x)

        x = x + x_256

        x = self.G_deconv2_3(x)
        x = self.G_deconv2_2(x)
        x = self.G_deconv2_1(x)
        x = self.G_deconv2_0(x)

        x = x + x_512

        x = self.G_deconv1_3(x)
        x = self.G_deconv1_2(x)
        x = self.G_deconv1_1(x)
        x = self.G_deconv1_0(x)
        output = F.sigmoid(x)
        ##＃end enhancement network ###

        ######copy channel to RGB
        # out_a =torch.unsqueeze(out[:,1,:,:],1)

        if self.learn_residual:
            output = input + output
            output = torch.clamp(output, min=-1, max=1)

        # output enhancement results
        # out segmentation result
        # fusion feature d3 d2 d1
        # d3 1/8 feature 128 channel
        # d2 1/4 feature 64 channel
        # d1 1/2 feature 64 channel
        return output, out


#################################

class Encode_Decode_seg_2scale_share(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Encode_Decode_seg_2scale_share, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.G_conv1 = nn.Conv2d(6, 32, 5, padding=2, bias=use_bias)
        self.G_relu1 = nn.ReLU()
        self.G_conv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv2_0 = nn.Conv2d(32, 64, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu2_0 = nn.ReLU()
        # concat 1/2
        self.G_conv2 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
        self.G_relu2 = nn.ReLU()
        self.G_conv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv3_0 = nn.Conv2d(64, 64, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu3_0 = nn.ReLU()
        # concat 1/4
        self.G_conv3 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
        self.G_relu3 = nn.ReLU()
        self.G_conv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv4_0 = nn.Conv2d(64, 128, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu4_0 = nn.ReLU()
        # concat 1/8
        self.G_conv4 = nn.Conv2d(256, 128, 5, padding=2, bias=use_bias)
        self.G_relu4 = nn.ReLU()
        self.G_conv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_deconv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_0 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=use_bias)

        self.G_deconv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_0 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=use_bias)

        self.G_deconv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_0 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=use_bias)

        self.G_deconv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_0 = nn.Conv2d(32, 3, 5, padding=2, bias=use_bias)

        self.G_pool_256 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.G_pool_128 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.G_pool_64 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.G_input_2 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=use_bias)

        #### CE_Net ######

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])

        self.decoder2 = DecoderBlock(filters[1], filters[0])

        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, input_512, input_256, input_norm):

        ### begin segmentation network ##
        x = self.firstconv(input_norm)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        out = F.sigmoid(out)

        ###　end segmentation network ###
        ### begin enhance network ###
        d1_1 = self.G_pool_256(d1)
        d2_1 = self.G_pool_128(d2)
        d3_1 = self.G_pool_64(d3)

        input_copy_256 = torch.cat([input_256, input_256], 1)
        x = self.G_conv1(input_copy_256)
        x = self.G_relu1(x)
        x = self.G_conv1_1(x)
        x = self.G_conv1_2(x)
        x = self.G_conv1_3(x)

        x = self.G_conv2_0(x)
        x = self.G_relu2_0(x)
        con_2 = torch.cat([x, d1_1], 1)
        x = self.G_conv2(con_2)
        x = self.G_relu2(x)
        x = self.G_conv2_1(x)
        x = self.G_conv2_2(x)
        x = self.G_conv2_３(x)

        x = self.G_conv3_0(x)
        x = self.G_relu３_0(x)
        con_4 = torch.cat([x, d2_1], 1)
        x = self.G_conv3(con_4)
        x = self.G_relu3(x)
        x = self.G_conv3_1(x)
        x = self.G_conv3_2(x)
        x = self.G_conv3_3(x)

        x = self.G_conv4_0(x)
        x = self.G_relu４_0(x)
        con_8 = torch.cat([x, d3_1], 1)
        x = self.G_conv4(con_8)
        x = self.G_relu4(x)
        x = self.G_conv4_1(x)
        x = self.G_conv4_2(x)
        x = self.G_conv4_3(x)

        x = self.G_deconv4_3(x)
        x = self.G_deconv4_2(x)
        x = self.G_deconv4_1(x)
        x = self.G_deconv4_0(x)

        x = self.G_deconv3_3(x)
        x = self.G_deconv3_2(x)
        x = self.G_deconv3_1(x)
        x = self.G_deconv3_0(x)

        x = self.G_deconv2_3(x)
        x = self.G_deconv2_2(x)
        x = self.G_deconv2_1(x)
        x = self.G_deconv2_0(x)

        x = self.G_deconv1_3(x)
        x = self.G_deconv1_2(x)
        x = self.G_deconv1_1(x)
        x = self.G_deconv1_0(x)
        output_256 = F.sigmoid(x)
        input_2 = self.G_input_2(output_256)

        # ori_scale
        input_copy_512 = torch.cat([input_512, input_2], 1)
        x = self.G_conv1(input_copy_512)
        x = self.G_relu1(x)
        x = self.G_conv1_1(x)
        x = self.G_conv1_2(x)
        x = self.G_conv1_3(x)

        x = self.G_conv2_0(x)
        x = self.G_relu2_0(x)
        con_2 = torch.cat([x, d1], 1)
        x = self.G_conv2(con_2)
        x = self.G_relu2(x)
        x = self.G_conv2_1(x)
        x = self.G_conv2_2(x)
        x = self.G_conv2_３(x)

        x = self.G_conv3_0(x)
        x = self.G_relu３_0(x)
        con_4 = torch.cat([x, d2], 1)
        x = self.G_conv3(con_4)
        x = self.G_relu3(x)
        x = self.G_conv3_1(x)
        x = self.G_conv3_2(x)
        x = self.G_conv3_3(x)

        x = self.G_conv4_0(x)
        x = self.G_relu４_0(x)
        con_8 = torch.cat([x, d3], 1)
        x = self.G_conv4(con_8)
        x = self.G_relu4(x)
        x = self.G_conv4_1(x)
        x = self.G_conv4_2(x)
        x = self.G_conv4_3(x)

        x = self.G_deconv4_3(x)
        x = self.G_deconv4_2(x)
        x = self.G_deconv4_1(x)
        x = self.G_deconv4_0(x)

        x = self.G_deconv3_3(x)
        x = self.G_deconv3_2(x)
        x = self.G_deconv3_1(x)
        x = self.G_deconv3_0(x)

        x = self.G_deconv2_3(x)
        x = self.G_deconv2_2(x)
        x = self.G_deconv2_1(x)
        x = self.G_deconv2_0(x)

        x = self.G_deconv1_3(x)
        x = self.G_deconv1_2(x)
        x = self.G_deconv1_1(x)
        x = self.G_deconv1_0(x)
        output_512 = F.sigmoid(x)
        ##＃end enhancement network ###

        ######copy channel to RGB
        # out_a =torch.unsqueeze(out[:,1,:,:],1)

        # output enhancement results
        # out segmentation result
        # fusion feature d3 d2 d1
        # d3 1/8 feature 128 channel
        # d2 1/4 feature 64 channel
        # d1 1/2 feature 64 channel
        return output_512, output_256, out


class Encode_Decode_seg_2scale_share_res(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Encode_Decode_seg_2scale_share_res, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.G_conv1 = nn.Conv2d(6, 32, 5, padding=2, bias=use_bias)
        self.G_relu1 = nn.ReLU()
        self.G_conv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv2_0 = nn.Conv2d(32, 64, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu2_0 = nn.ReLU()
        # concat 1/2
        self.G_conv2 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
        self.G_relu2 = nn.ReLU()
        self.G_conv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv3_0 = nn.Conv2d(64, 64, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu3_0 = nn.ReLU()
        # concat 1/4
        self.G_conv3 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
        self.G_relu3 = nn.ReLU()
        self.G_conv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv4_0 = nn.Conv2d(64, 128, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu4_0 = nn.ReLU()
        # concat 1/8
        self.G_conv4 = nn.Conv2d(256, 128, 5, padding=2, bias=use_bias)
        self.G_relu4 = nn.ReLU()
        self.G_conv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_deconv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_0 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=use_bias)

        self.G_deconv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_0 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=use_bias)

        self.G_deconv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_0 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=use_bias)

        self.G_deconv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_0 = nn.Conv2d(32, 3, 5, padding=2, bias=use_bias)

        self.G_pool_256 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.G_pool_128 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.G_pool_64 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.G_input_2 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=use_bias)
        #### CE_Net ######

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])

        self.decoder2 = DecoderBlock(filters[1], filters[0])

        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, input_512, input_256, input_norm):

        ### begin segmentation network ###
        x = self.firstconv(input_norm)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        out = F.sigmoid(out)

        ###　end segmentation network ###

        ### begin enhance network ###
        d1_1 = self.G_pool_256(d1)
        d2_1 = self.G_pool_128(d2)
        d3_1 = self.G_pool_64(d3)

        input_copy_256 = torch.cat([input_256, input_256], 1)
        x = self.G_conv1(input_copy_256)
        x = self.G_relu1(x)
        x = self.G_conv1_1(x)
        x = self.G_conv1_2(x)
        x_512 = self.G_conv1_3(x)

        x = self.G_conv2_0(x_512)
        x = self.G_relu2_0(x)
        con_2 = torch.cat([x, d1_1], 1)
        x = self.G_conv2(con_2)
        x = self.G_relu2(x)
        x = self.G_conv2_1(x)
        x = self.G_conv2_2(x)
        x_256 = self.G_conv2_３(x)

        x = self.G_conv3_0(x_256)
        x = self.G_relu3_0(x)
        con_4 = torch.cat([x, d2_1], 1)
        x = self.G_conv3(con_4)
        x = self.G_relu3(x)
        x = self.G_conv3_1(x)
        x = self.G_conv3_2(x)
        x_128 = self.G_conv3_3(x)

        x = self.G_conv4_0(x_128)
        x = self.G_relu4_0(x)
        con_8 = torch.cat([x, d3_1], 1)
        x = self.G_conv4(con_8)
        x = self.G_relu4(x)
        x = self.G_conv4_1(x)
        x = self.G_conv4_2(x)
        x = self.G_conv4_3(x)

        x = self.G_deconv4_3(x)
        x = self.G_deconv4_2(x)
        x = self.G_deconv4_1(x)
        x = self.G_deconv4_0(x)

        x = x + x_128

        x = self.G_deconv3_3(x)
        x = self.G_deconv3_2(x)
        x = self.G_deconv3_1(x)
        x = self.G_deconv3_0(x)

        x = x + x_256

        x = self.G_deconv2_3(x)
        x = self.G_deconv2_2(x)
        x = self.G_deconv2_1(x)
        x = self.G_deconv2_0(x)

        x = x + x_512

        x = self.G_deconv1_3(x)
        x = self.G_deconv1_2(x)
        x = self.G_deconv1_1(x)
        x = self.G_deconv1_0(x)
        output_256 = F.sigmoid(x)
        input_2 = self.G_input_2(output_256)

        # ori_scale
        input_copy_512 = torch.cat([input_512, input_2], 1)
        x = self.G_conv1(input_copy_512)
        x = self.G_relu1(x)
        x = self.G_conv1_1(x)
        x = self.G_conv1_2(x)
        x_512 = self.G_conv1_3(x)

        x = self.G_conv2_0(x_512)
        x = self.G_relu2_0(x)
        con_2 = torch.cat([x, d1], 1)
        x = self.G_conv2(con_2)
        x = self.G_relu2(x)
        x = self.G_conv2_1(x)
        x = self.G_conv2_2(x)
        x_256 = self.G_conv2_３(x)

        x = self.G_conv3_0(x_256)
        x = self.G_relu3_0(x)
        con_4 = torch.cat([x, d2], 1)
        x = self.G_conv3(con_4)
        x = self.G_relu3(x)
        x = self.G_conv3_1(x)
        x = self.G_conv3_2(x)
        x_128 = self.G_conv3_3(x)

        x = self.G_conv4_0(x_128)
        x = self.G_relu4_0(x)
        con_8 = torch.cat([x, d3], 1)
        x = self.G_conv4(con_8)
        x = self.G_relu4(x)
        x = self.G_conv4_1(x)
        x = self.G_conv4_2(x)
        x = self.G_conv4_3(x)

        x = self.G_deconv4_3(x)
        x = self.G_deconv4_2(x)
        x = self.G_deconv4_1(x)
        x = self.G_deconv4_0(x)

        x = x + x_128

        x = self.G_deconv3_3(x)
        x = self.G_deconv3_2(x)
        x = self.G_deconv3_1(x)
        x = self.G_deconv3_0(x)

        x = x + x_256

        x = self.G_deconv2_3(x)
        x = self.G_deconv2_2(x)
        x = self.G_deconv2_1(x)
        x = self.G_deconv2_0(x)

        x = x + x_512

        x = self.G_deconv1_3(x)
        x = self.G_deconv1_2(x)
        x = self.G_deconv1_1(x)
        x = self.G_deconv1_0(x)
        output_512 = F.sigmoid(x)

        ##＃end enhancement network ###

        ######copy channel to RGB
        # out_a =torch.unsqueeze(out[:,1,:,:],1)

        if self.learn_residual:
            output = input + output
            output = torch.clamp(output, min=-1, max=1)

        # output enhancement results
        # out segmentation result
        # fusion feature d3 d2 d1
        # d3 1/8 feature 128 channel
        # d2 1/4 feature 64 channel
        # d1 1/2 feature 64 channel
        return output_512, output_256, out


class Encode_Decode_seg_2scale_share_atten_res_sum(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Encode_Decode_seg_2scale_share_atten_res_sum, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.G_conv1 = nn.Conv2d(6, 32, 5, padding=2, bias=use_bias)
        self.G_relu1 = nn.ReLU()
        self.G_conv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv2_0 = nn.Conv2d(32, 64, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu2_0 = nn.ReLU()
        # concat 1/2
        self.G_conv2 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
        self.G_relu2 = nn.ReLU()
        self.G_conv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv3_0 = nn.Conv2d(64, 64, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu3_0 = nn.ReLU()
        # concat 1/4
        self.G_conv3 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
        self.G_relu3 = nn.ReLU()
        self.G_conv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv4_0 = nn.Conv2d(64, 128, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu4_0 = nn.ReLU()
        # concat 1/8
        self.G_conv4 = nn.Conv2d(256, 128, 5, padding=2, bias=use_bias)
        self.G_relu4 = nn.ReLU()
        self.G_conv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_deconv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_0 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=use_bias)

        self.G_deconv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_0 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=use_bias)

        self.G_deconv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_0 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=use_bias)

        self.G_deconv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_0 = nn.Conv2d(32, 3, 5, padding=2, bias=use_bias)

        self.G_pool_256 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.G_pool_128 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.G_pool_64 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.G_input_2 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=use_bias)

        #### CE_Net #####
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])

        self.decoder2 = DecoderBlock(filters[1], filters[0])

        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)

        #### begin attention module ####
        self.a_in_pool = nn.MaxPool2d(kernel_size=[2, 2], stride=2)

        self.a_en_conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=use_bias)
        self.a_en_relu1 = nn.ReLU()
        self.a_en_pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)

        self.a_en_conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=use_bias)
        self.a_en_relu2 = nn.ReLU()
        self.a_en_pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)

        self.a_en_conv3 = nn.Conv2d(128, 256, 3, padding=1, bias=use_bias)
        self.a_en_relu3 = nn.ReLU()
        self.a_en_pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)

        self.a_de_conv3 = nn.Conv2d(256, 256, 3, padding=1, bias=use_bias)
        self.a_de_relu31 = nn.ReLU()
        self.a_de_deconv3 = nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=use_bias)
        self.a_de_relu32 = nn.ReLU()

        self.a_de_conv2 = nn.Conv2d(256, 128, 3, padding=1, bias=use_bias)
        self.a_de_relu21 = nn.ReLU()
        self.a_de_deconv2 = nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=use_bias)
        self.a_de_relu22 = nn.ReLU()

        self.a_de_conv1 = nn.Conv2d(128, 64, 3, padding=1, bias=use_bias)
        self.a_de_relu11 = nn.ReLU()
        self.a_de_deconv1 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=use_bias)
        self.a_de_relu12 = nn.ReLU()

        self.a_spot_256 = nn.Conv2d(64, 1, 1, padding=0, bias=use_bias)
        self.a_pool_128 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)

    def forward(self, input_512, input_256, input_norm):

        ### begin segmentation network ###
        x = self.firstconv(input_norm)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        out = F.sigmoid(out)

        ###　end segmentation network ###
        ### begin attention network ###
        x = self.a_in_pool(input_512)
        x = self.a_en_conv1(x)
        x = self.a_en_relu1(x)
        x = self.a_en_pool1(x)

        x = self.a_en_conv2(x)
        x = self.a_en_relu2(x)
        x = self.a_en_pool2(x)

        x = self.a_en_conv3(x)
        x = self.a_en_relu3(x)
        x = self.a_en_pool3(x)

        x = self.a_de_conv3(x)
        x = self.a_de_relu31(x)
        x = self.a_de_deconv3(x)
        x = self.a_de_relu32(x)

        x = self.a_de_conv2(x)
        x = self.a_de_relu21(x)
        x = self.a_de_deconv2(x)
        x = self.a_de_relu22(x)

        x = self.a_de_conv1(x)
        x = self.a_de_relu11(x)
        x = self.a_de_deconv1(x)
        x = self.a_de_relu12(x)

        a_mask = self.a_spot_256(x)

        ### end attention network ###
        ### begin enhance network ###
        d1_1 = self.G_pool_256(d1)
        d2_1 = self.G_pool_128(d2)
        d3_1 = self.G_pool_64(d3)
        a_mask_1 = self.a_pool_128(a_mask)

        input_copy_256 = torch.cat([input_256, input_256], 1)
        x = self.G_conv1(input_copy_256)
        x = self.G_relu1(x)
        x = self.G_conv1_1(x)
        x = self.G_conv1_2(x)
        x_512 = self.G_conv1_3(x)

        x = self.G_conv2_0(x_512)
        x = self.G_relu2_0(x)
        x = x * a_mask_1.expand_as(x)  # +x
        # con_2 = x +d1_1
        con_2 = torch.cat([x, d1_1], 1)
        x = self.G_conv2(con_2)
        x = self.G_relu2(x)
        x = self.G_conv2_1(x)
        x = self.G_conv2_2(x)
        x_256 = self.G_conv2_３(x)

        x = self.G_conv3_0(x_256)
        x = self.G_relu3_0(x)
        con_4 = torch.cat([x, d2_1], 1)
        x = self.G_conv3(con_4)
        x = self.G_relu3(x)
        x = self.G_conv3_1(x)
        x = self.G_conv3_2(x)
        x_128 = self.G_conv3_3(x)

        x = self.G_conv4_0(x_128)
        x = self.G_relu4_0(x)
        con_8 = torch.cat([x, d3_1], 1)
        x = self.G_conv4(con_8)
        x = self.G_relu4(x)
        x = self.G_conv4_1(x)
        x = self.G_conv4_2(x)
        x = self.G_conv4_3(x)

        x = self.G_deconv4_3(x)
        x = self.G_deconv4_2(x)
        x = self.G_deconv4_1(x)
        x = self.G_deconv4_0(x)

        x = x + x_128

        x = self.G_deconv3_3(x)
        x = self.G_deconv3_2(x)
        x = self.G_deconv3_1(x)
        x = self.G_deconv3_0(x)

        x = x + x_256

        x = self.G_deconv2_3(x)
        x = self.G_deconv2_2(x)
        x = self.G_deconv2_1(x)
        x = self.G_deconv2_0(x)

        x = x + x_512

        x = self.G_deconv1_3(x)
        x = self.G_deconv1_2(x)
        x = self.G_deconv1_1(x)
        x = self.G_deconv1_0(x)
        output_256 = F.sigmoid(x)
        input_2 = self.G_input_2(output_256)

        # ori_scale
        input_copy_512 = torch.cat([input_512, input_2], 1)
        x = self.G_conv1(input_copy_512)
        x = self.G_relu1(x)
        x = self.G_conv1_1(x)
        x = self.G_conv1_2(x)
        x_512 = self.G_conv1_3(x)

        x = self.G_conv2_0(x_512)
        x = self.G_relu2_0(x)
        x = x * a_mask.expand_as(x)  # +x
        # con_2 = x + d1
        con_2 = torch.cat([x, d1], 1)
        x = self.G_conv2(con_2)
        x = self.G_relu2(x)
        x = self.G_conv2_1(x)
        x = self.G_conv2_2(x)
        x_256 = self.G_conv2_３(x)

        x = self.G_conv3_0(x_256)
        x = self.G_relu3_0(x)
        con_4 = torch.cat([x, d2], 1)
        x = self.G_conv3(con_4)
        x = self.G_relu3(x)
        x = self.G_conv3_1(x)
        x = self.G_conv3_2(x)
        x_128 = self.G_conv3_3(x)

        x = self.G_conv4_0(x_128)
        x = self.G_relu4_0(x)
        con_8 = torch.cat([x, d3], 1)
        x = self.G_conv4(con_8)
        x = self.G_relu4(x)
        x = self.G_conv4_1(x)
        x = self.G_conv4_2(x)
        x = self.G_conv4_3(x)

        x = self.G_deconv4_3(x)
        x = self.G_deconv4_2(x)
        x = self.G_deconv4_1(x)
        x = self.G_deconv4_0(x)

        x = x + x_128

        x = self.G_deconv3_3(x)
        x = self.G_deconv3_2(x)
        x = self.G_deconv3_1(x)
        x = self.G_deconv3_0(x)

        x = x + x_256

        x = self.G_deconv2_3(x)
        x = self.G_deconv2_2(x)
        x = self.G_deconv2_1(x)
        x = self.G_deconv2_0(x)

        x = x + x_512

        x = self.G_deconv1_3(x)
        x = self.G_deconv1_2(x)
        x = self.G_deconv1_1(x)
        x = self.G_deconv1_0(x)
        output_512 = F.sigmoid(x)

        ##＃end enhancement network ###

        # output enhancement results
        # out segmentation result
        # fusion feature d3 d2 d1
        # d3 1/8 feature 128 channel
        # d2 1/4 feature 64 channel
        # d1 1/2 feature 64 channel
        return output_512, output_256, out, a_mask


#  concat
class Encode_Decode_seg_2scale_share_atten_res(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Encode_Decode_seg_2scale_share_atten_res, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.G_conv1 = nn.Conv2d(6, 32, 5, padding=2, bias=use_bias)
        self.G_relu1 = nn.ReLU()
        self.G_conv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv2_0 = nn.Conv2d(32, 64, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu2_0 = nn.ReLU()
        # concat 1/2
        self.G_conv2 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
        self.G_relu2 = nn.ReLU()
        self.G_conv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv3_0 = nn.Conv2d(64, 64, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu3_0 = nn.ReLU()
        # concat 1/4
        self.G_conv3 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
        self.G_relu3 = nn.ReLU()
        self.G_conv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv4_0 = nn.Conv2d(64, 128, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu4_0 = nn.ReLU()
        # concat 1/8
        self.G_conv4 = nn.Conv2d(256, 128, 5, padding=2, bias=use_bias)
        self.G_relu4 = nn.ReLU()
        self.G_conv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_deconv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_0 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=use_bias)

        self.G_deconv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_0 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=use_bias)

        self.G_deconv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_0 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=use_bias)

        self.G_deconv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_0 = nn.Conv2d(32, 3, 5, padding=2, bias=use_bias)

        self.G_pool_256 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.G_pool_128 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.G_pool_64 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.G_input_2 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=use_bias)

        #### CE_Net #####
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])

        self.decoder2 = DecoderBlock(filters[1], filters[0])

        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)

        #### begin attention module ####
        self.a_in_pool = nn.MaxPool2d(kernel_size=[2, 2], stride=2)

        self.a_en_conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=use_bias)
        self.a_en_relu1 = nn.ReLU()
        self.a_en_pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)

        self.a_en_conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=use_bias)
        self.a_en_relu2 = nn.ReLU()
        self.a_en_pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)

        self.a_en_conv3 = nn.Conv2d(128, 256, 3, padding=1, bias=use_bias)
        self.a_en_relu3 = nn.ReLU()
        self.a_en_pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)

        self.a_de_conv3 = nn.Conv2d(256, 256, 3, padding=1, bias=use_bias)
        self.a_de_relu31 = nn.ReLU()
        self.a_de_deconv3 = nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=use_bias)
        self.a_de_relu32 = nn.ReLU()

        self.a_de_conv2 = nn.Conv2d(256, 128, 3, padding=1, bias=use_bias)
        self.a_de_relu21 = nn.ReLU()
        self.a_de_deconv2 = nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=use_bias)
        self.a_de_relu22 = nn.ReLU()

        self.a_de_conv1 = nn.Conv2d(128, 64, 3, padding=1, bias=use_bias)
        self.a_de_relu11 = nn.ReLU()
        self.a_de_deconv1 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=use_bias)
        self.a_de_relu12 = nn.ReLU()

        self.a_spot_256 = nn.Conv2d(64, 1, 1, padding=0, bias=use_bias)
        self.a_pool_128 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)

    def forward(self, input_512, input_256, input_norm):

        ### begin segmentation network ###
        x = self.firstconv(input_norm)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        out = F.sigmoid(out)

        ###　end segmentation network ###
        ### begin attention network ###
        x = self.a_in_pool(input_512)
        x = self.a_en_conv1(x)
        x = self.a_en_relu1(x)
        x = self.a_en_pool1(x)

        x = self.a_en_conv2(x)
        x = self.a_en_relu2(x)
        x = self.a_en_pool2(x)

        x = self.a_en_conv3(x)
        x = self.a_en_relu3(x)
        x = self.a_en_pool3(x)

        x = self.a_de_conv3(x)
        x = self.a_de_relu31(x)
        x = self.a_de_deconv3(x)
        x = self.a_de_relu32(x)

        x = self.a_de_conv2(x)
        x = self.a_de_relu21(x)
        x = self.a_de_deconv2(x)
        x = self.a_de_relu22(x)

        x = self.a_de_conv1(x)
        x = self.a_de_relu11(x)
        x = self.a_de_deconv1(x)
        x = self.a_de_relu12(x)

        a_mask = self.a_spot_256(x)

        ### end attention network ###
        ### begin enhance network ###
        d1_1 = self.G_pool_256(d1)
        d2_1 = self.G_pool_128(d2)
        d3_1 = self.G_pool_64(d3)
        a_mask_1 = self.a_pool_128(a_mask)

        input_copy_256 = torch.cat([input_256, input_256], 1)
        x = self.G_conv1(input_copy_256)
        x = self.G_relu1(x)
        x = self.G_conv1_1(x)
        x = self.G_conv1_2(x)
        x_512 = self.G_conv1_3(x)

        x = self.G_conv2_0(x_512)
        x = self.G_relu2_0(x)
        x = x * a_mask_1.expand_as(x) + x
        con_2 = torch.cat([x, d1_1], 1)
        x = self.G_conv2(con_2)
        x = self.G_relu2(x)
        x = self.G_conv2_1(x)
        x = self.G_conv2_2(x)
        x_256 = self.G_conv2_３(x)

        x = self.G_conv3_0(x_256)
        x = self.G_relu3_0(x)
        con_4 = torch.cat([x, d2_1], 1)
        x = self.G_conv3(con_4)
        x = self.G_relu3(x)
        x = self.G_conv3_1(x)
        x = self.G_conv3_2(x)
        x_128 = self.G_conv3_3(x)

        x = self.G_conv4_0(x_128)
        x = self.G_relu4_0(x)
        con_8 = torch.cat([x, d3_1], 1)
        x = self.G_conv4(con_8)
        x = self.G_relu4(x)
        x = self.G_conv4_1(x)
        x = self.G_conv4_2(x)
        x = self.G_conv4_3(x)

        x = self.G_deconv4_3(x)
        x = self.G_deconv4_2(x)
        x = self.G_deconv4_1(x)
        x = self.G_deconv4_0(x)

        x = x + x_128

        x = self.G_deconv3_3(x)
        x = self.G_deconv3_2(x)
        x = self.G_deconv3_1(x)
        x = self.G_deconv3_0(x)

        x = x + x_256

        x = self.G_deconv2_3(x)
        x = self.G_deconv2_2(x)
        x = self.G_deconv2_1(x)
        x = self.G_deconv2_0(x)

        x = x + x_512

        x = self.G_deconv1_3(x)
        x = self.G_deconv1_2(x)
        x = self.G_deconv1_1(x)
        x = self.G_deconv1_0(x)
        output_256 = F.sigmoid(x)
        input_2 = self.G_input_2(output_256)

        # ori_scale
        input_copy_512 = torch.cat([input_512, input_2], 1)
        x = self.G_conv1(input_copy_512)
        x = self.G_relu1(x)
        x = self.G_conv1_1(x)
        x = self.G_conv1_2(x)
        x_512 = self.G_conv1_3(x)

        x = self.G_conv2_0(x_512)
        x = self.G_relu2_0(x)
        x = x * a_mask.expand_as(x) + x
        con_2 = torch.cat([x, d1], 1)
        x = self.G_conv2(con_2)
        x = self.G_relu2(x)
        x = self.G_conv2_1(x)
        x = self.G_conv2_2(x)
        x_256 = self.G_conv2_３(x)

        x = self.G_conv3_0(x_256)
        x = self.G_relu3_0(x)
        con_4 = torch.cat([x, d2], 1)
        x = self.G_conv3(con_4)
        x = self.G_relu3(x)
        x = self.G_conv3_1(x)
        x = self.G_conv3_2(x)
        x_128 = self.G_conv3_3(x)

        x = self.G_conv4_0(x_128)
        x = self.G_relu4_0(x)
        con_8 = torch.cat([x, d3], 1)
        x = self.G_conv4(con_8)
        x = self.G_relu4(x)
        x = self.G_conv4_1(x)
        x = self.G_conv4_2(x)
        x = self.G_conv4_3(x)

        x = self.G_deconv4_3(x)
        x = self.G_deconv4_2(x)
        x = self.G_deconv4_1(x)
        x = self.G_deconv4_0(x)

        x = x + x_128

        x = self.G_deconv3_3(x)
        x = self.G_deconv3_2(x)
        x = self.G_deconv3_1(x)
        x = self.G_deconv3_0(x)

        x = x + x_256

        x = self.G_deconv2_3(x)
        x = self.G_deconv2_2(x)
        x = self.G_deconv2_1(x)
        x = self.G_deconv2_0(x)

        x = x + x_512

        x = self.G_deconv1_3(x)
        x = self.G_deconv1_2(x)
        x = self.G_deconv1_1(x)
        x = self.G_deconv1_0(x)
        output_512 = F.sigmoid(x)

        ##＃end enhancement network ###

        # output enhancement results
        # out segmentation result
        # fusion feature d3 d2 d1
        # d3 1/8 feature 128 channel
        # d2 1/4 feature 64 channel
        # d1 1/2 feature 64 channel
        return output_512, output_256, out, a_mask


# ########################################################
class Encode_Decode(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Encode_Decode, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.G_conv1 = nn.Conv2d(3, 32, 5, padding=2, bias=use_bias)
        self.G_relu1 = nn.ReLU()
        self.G_conv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv2_0 = nn.Conv2d(32, 64, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu2_0 = nn.ReLU()
        # concat 1/2
        self.G_conv2 = nn.Conv2d(64, 64, 5, padding=2, bias=use_bias)
        self.G_relu2 = nn.ReLU()
        self.G_conv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv3_0 = nn.Conv2d(64, 64, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu3_0 = nn.ReLU()
        # concat 1/4
        self.G_conv3 = nn.Conv2d(64, 64, 5, padding=2, bias=use_bias)
        self.G_relu3 = nn.ReLU()
        self.G_conv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv4_0 = nn.Conv2d(64, 128, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu4_0 = nn.ReLU()
        # concat 1/8
        self.G_conv4 = nn.Conv2d(128, 128, 5, padding=2, bias=use_bias)
        self.G_relu４ = nn.ReLU()
        self.G_conv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_deconv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_0 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=use_bias)

        self.G_deconv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_0 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=use_bias)

        self.G_deconv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_0 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=use_bias)

        self.G_deconv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_0 = nn.Conv2d(32, 3, 5, padding=2, bias=use_bias)
        self.G_final = nn.Sigmoid()

    def forward(self, input):

        ### begin enhance network ###

        x = self.G_conv1(input)
        x = self.G_relu1(x)
        x = self.G_conv1_1(x)
        x = self.G_conv1_2(x)
        x = self.G_conv1_3(x)

        x = self.G_conv2_0(x)
        x = self.G_relu2_0(x)
        # con_2 = torch.cat([x,d1],1)
        x = self.G_conv2(x)
        x = self.G_relu2(x)
        x = self.G_conv2_1(x)
        x = self.G_conv2_2(x)
        x = self.G_conv2_３(x)

        x = self.G_conv3_0(x)
        x = self.G_relu3_0(x)
        # con_4 = torch.cat([x,d2],1)
        x = self.G_conv3(x)
        x = self.G_relu3(x)
        x = self.G_conv3_1(x)
        x = self.G_conv3_2(x)
        x = self.G_conv3_3(x)

        x = self.G_conv4_0(x)
        x = self.G_relu４_0(x)
        # con_8 = torch.cat([x,d3],1)
        x = self.G_conv4(x)
        x = self.G_relu4(x)
        x = self.G_conv4_1(x)
        x = self.G_conv4_2(x)
        x = self.G_conv4_3(x)

        x = self.G_deconv4_3(x)
        x = self.G_deconv4_2(x)
        x = self.G_deconv4_1(x)
        x = self.G_deconv4_0(x)

        x = self.G_deconv3_3(x)
        x = self.G_deconv3_2(x)
        x = self.G_deconv3_1(x)
        x = self.G_deconv3_0(x)

        x = self.G_deconv2_3(x)
        x = self.G_deconv2_2(x)
        x = self.G_deconv2_1(x)
        x = self.G_deconv2_0(x)

        x = self.G_deconv1_3(x)
        x = self.G_deconv1_2(x)
        x = self.G_deconv1_1(x)
        x = self.G_deconv1_0(x)
        output = self.G_final(x)
        # output = F.sigmoid(x)

        return output


class Encode_Decode_res(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Encode_Decode_res, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.G_conv1 = nn.Conv2d(3, 32, 5, padding=2, bias=use_bias)
        self.G_relu1 = nn.ReLU()
        self.G_conv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv2_0 = nn.Conv2d(32, 64, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu2_0 = nn.ReLU()
        # concat 1/2
        self.G_conv2 = nn.Conv2d(64, 64, 5, padding=2, bias=use_bias)
        self.G_relu2 = nn.ReLU()
        self.G_conv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv3_0 = nn.Conv2d(64, 64, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu3_0 = nn.ReLU()
        # concat 1/4
        self.G_conv3 = nn.Conv2d(64, 64, 5, padding=2, bias=use_bias)
        self.G_relu3 = nn.ReLU()
        self.G_conv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_conv4_0 = nn.Conv2d(64, 128, 5, padding=2, stride=2, bias=use_bias)
        self.G_relu4_0 = nn.ReLU()
        # concat 1/8
        self.G_conv4 = nn.Conv2d(128, 128, 5, padding=2, bias=use_bias)
        self.G_relu4 = nn.ReLU()
        self.G_conv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_conv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)

        self.G_deconv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv4_0 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=use_bias)

        self.G_deconv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv3_0 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=use_bias)

        self.G_deconv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv2_0 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=use_bias)

        self.G_deconv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
        self.G_deconv1_0 = nn.Conv2d(32, 3, 5, padding=2, bias=use_bias)

    def forward(self, input):

        x = self.G_conv1(input)
        x = self.G_relu1(x)
        x = self.G_conv1_1(x)
        x = self.G_conv1_2(x)
        x_512 = self.G_conv1_3(x)

        x = self.G_conv2_0(x_512)
        x = self.G_relu2_0(x)
        # con_2 = torch.cat([x,d1],1)
        x = self.G_conv2(x)
        x = self.G_relu2(x)
        x = self.G_conv2_1(x)
        x = self.G_conv2_2(x)
        x_256 = self.G_conv2_３(x)

        x = self.G_conv3_0(x_256)
        x = self.G_relu3_0(x)
        # con_4 = torch.cat([x,d2],1)
        x = self.G_conv3(x)
        x = self.G_relu3(x)
        x = self.G_conv3_1(x)
        x = self.G_conv3_2(x)
        x_128 = self.G_conv3_3(x)

        x = self.G_conv4_0(x_128)
        x = self.G_relu4_0(x)
        # con_8 = torch.cat([x,d3],1)
        x = self.G_conv4(x)
        x = self.G_relu4(x)
        x = self.G_conv4_1(x)
        x = self.G_conv4_2(x)
        x = self.G_conv4_3(x)

        x = self.G_deconv4_3(x)
        x = self.G_deconv4_2(x)
        x = self.G_deconv4_1(x)
        x = self.G_deconv4_0(x)

        x = x + x_128

        x = self.G_deconv3_3(x)
        x = self.G_deconv3_2(x)
        x = self.G_deconv3_1(x)
        x = self.G_deconv3_0(x)

        x = x + x_256

        x = self.G_deconv2_3(x)
        x = self.G_deconv2_2(x)
        x = self.G_deconv2_1(x)
        x = self.G_deconv2_0(x)

        x = x + x_512

        x = self.G_deconv1_3(x)
        x = self.G_deconv1_2(x)
        x = self.G_deconv1_1(x)
        x = self.G_deconv1_0(x)
        output = F.sigmoid(x)
        ##＃end enhancement network ###
        x = self.G_conv1(output)
        x = self.G_relu1(x)
        x = self.G_conv1_1(x)
        x = self.G_conv1_2(x)
        x_512 = self.G_conv1_3(x)

        x = self.G_conv2_0(x_512)
        x = self.G_relu2_0(x)
        # con_2 = torch.cat([x,d1],1)
        x = self.G_conv2(x)
        x = self.G_relu2(x)
        x = self.G_conv2_1(x)
        x = self.G_conv2_2(x)
        x_256 = self.G_conv2_３(x)

        x = self.G_conv3_0(x_256)
        x = self.G_relu3_0(x)
        # con_4 = torch.cat([x,d2],1)
        x = self.G_conv3(x)
        x = self.G_relu3(x)
        x = self.G_conv3_1(x)
        x = self.G_conv3_2(x)
        x_128 = self.G_conv3_3(x)

        x = self.G_conv4_0(x_128)
        x = self.G_relu4_0(x)
        # con_8 = torch.cat([x,d3],1)
        x = self.G_conv4(x)
        x = self.G_relu4(x)
        x = self.G_conv4_1(x)
        x = self.G_conv4_2(x)
        x = self.G_conv4_3(x)

        x = self.G_deconv4_3(x)
        x = self.G_deconv4_2(x)
        x = self.G_deconv4_1(x)
        x = self.G_deconv4_0(x)

        x = x + x_128

        x = self.G_deconv3_3(x)
        x = self.G_deconv3_2(x)
        x = self.G_deconv3_1(x)
        x = self.G_deconv3_0(x)

        x = x + x_256

        x = self.G_deconv2_3(x)
        x = self.G_deconv2_2(x)
        x = self.G_deconv2_1(x)
        x = self.G_deconv2_0(x)

        x = x + x_512

        x = self.G_deconv1_3(x)
        x = self.G_deconv1_2(x)
        x = self.G_deconv1_1(x)
        x = self.G_deconv1_0(x)
        output = F.sigmoid(x)
        return output


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


####################################################
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], use_parallel=True, learn_residual=False):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        # currently support only input_nc == output_nc
        assert (input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer,
                                                 use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output, min=-1, max=1)
        return output


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[],
                 use_parallel=True):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


############################# vessel segmentation#########################################

class CE_Net(nn.Module):
    def __init__(self, num_classes=2, num_channels=3):
        super(CE_Net, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, in_kernel, in_pad, in_bias):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, in_kernel, padding=in_pad, bias=in_bias)
        self.relu1 = nonlinearity
        self.conv2 = nn.Conv2d(in_channels, in_channels, in_kernel, padding=in_pad, bias=in_bias)
        self.relu2 = nonlinearity

    def forward(self, x):
        x0 = self.conv1(x)
        x = self.relu2(x0)
        x = self.conv2(x)
        x = x0 + x
        out = self.relu2(x)
        return out
