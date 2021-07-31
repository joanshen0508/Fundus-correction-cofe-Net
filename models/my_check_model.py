import torch


if __name__ =='__main__':

    model_concat = torch.load('/home/szy/code/medical/TMI_EXP/modify_degrad_attention/encode_decode_vessel_addrelu_2scale_share_atten_res1_concatx/experiment_name/latest_net_G.pth')
    model_sum = torch.load('/home/szy/code/medical/TMI_EXP/modify_degrad_attention/encode_decode_vessel_addrelu_2scale_share_atten_res1_sumx/experiment_name/latest_net_G.pth')
    print('1')