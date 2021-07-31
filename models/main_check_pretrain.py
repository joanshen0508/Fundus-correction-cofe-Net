# import torch
# from torchvision import models
# from torchsummary import summary
#
# device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
# vgg =  models.vgg16().to(device)
#
# summary (vgg,3, 224, 224)

import torch
import torch.nn as nn

if __name__ =='__main__':
    pretrained_net = torch.load('/home/szy/code/medical/hdrcnn_imp/vgg16.pth')
    print(pretrained_net)
    for key, v in enumerate(pretrained_net):
        print (key, v)