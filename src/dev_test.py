from mindspore import nn
import mindspore as ms
from PIL import Image,ImageFile
from mindspore import Tensor
import numpy as np
from utils import *
import os
import json 
import networks
import mindspore_ssim

# new import
from mindspore import nn
import mindspore.experimental.optim as optim
from mindspore import context
from mindspore import save_checkpoint, load_checkpoint
from mindspore.ops import interpolate
from mindspore.nn import WithLossCell, TrainOneStepCell
from networks import vgg19
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from EnhanceN_arch import InteractNet

# 测试ssim效果
# 加载图片，转换为tensor
target = Image.open('/home/user/data/work/UHDFour_code-mindspore/data/UHD-LL/training_set/gt/4_UHD_LL.JPG').convert('RGB') #'input_unprocess_aligned/' v
final_result = Image.open('/home/user/data/work/UHDFour_code-mindspore/data/UHD-LL/training_set/input/4_UHD_LL.JPG').convert('RGB') #'input_unprocess_aligned/' v

# i1 = Tensor(np.expand_dims(np.transpose(np.array(i1), (2, 0, 1)), axis=0),ms.float32)
# i2 = Tensor(np.expand_dims(np.transpose(np.array(i2), (2, 0, 1)), axis=0),ms.float32)

target = Tensor(np.expand_dims(np.transpose(np.array(target), (2, 0, 1)), axis=0),ms.float32)
final_result = Tensor(np.expand_dims(np.transpose(np.array(final_result), (2, 0, 1)), axis=0),ms.float32)

context.set_context(device_target="GPU")
# ssim = nn.SSIM()
# print(ssim(i1, i2).mean())

# 测试VGG19模型参数迁移效果
# VGG = vgg19()
# vgg_path = "/home/user/data/work/UHDFour_code-mindspore/pre_trained_VGG19_modle/vgg19.ckpt"
# param_dict = load_checkpoint(vgg_path)
# load_param_into_net(VGG, param_dict)
# VGG.set_train(False)

# result_feature = VGG(interpolate(i1,mode='bilinear',size=(int(i1.shape[2]*0.05), int(i1.shape[3]*0.05))))
# target_feature = VGG(interpolate(i2,mode='bilinear',size=(int(i2.shape[2]*0.05), int(i2.shape[3]*0.05))))

# loss_per = 0.001*nn.MSELoss()(result_feature, target_feature) 

# print(loss_per)

# 打印模型参数
# from EnhanceN_arch import InteractNet

# def mindspore_params(network):
#     ms_params = {}
#     for param in network.get_parameters():
#         name = param.name
#         value = param.data.asnumpy()
#         print(name, value.shape)
#         ms_params[name] = value
#     return ms_params

# mindspore_params(InteractNet())

# 测试损失函数
# load VGG19 function
# VGG = vgg19()
# vgg_path = "/home/user/data/work/UHDFour_code-mindspore/pre_trained_VGG19_modle/vgg19.ckpt"
# param_dict = load_checkpoint(vgg_path)
# load_param_into_net(VGG, param_dict)
# VGG.set_train(False)
# # Loss function
# scale = 0.125

# L1 = nn.SmoothL1Loss(reduction='mean')
# L1down = nn.SmoothL1Loss(reduction='mean')
# L2 = nn.MSELoss()
# ssim = nn.SSIM()
# final_result_down = interpolate(final_result,mode='bilinear',size=(int(final_result.shape[2]*scale), int(final_result.shape[3]*scale)))


# loss_l1 = 5*L1(final_result,target) 

# loss_l1down = 0.5*L1(final_result_down, interpolate(target,mode='bilinear',size=(int(target.shape[2]*scale), int(target.shape[3]*scale)))) 
# result_feature = VGG(final_result_down)
# target_feature = VGG(interpolate(target,mode='bilinear',size=(int(target.shape[2]*scale), int(target.shape[3]*scale))))
# loss_per = 0.001*L2(result_feature, target_feature) 
# loss_ssim=0.002*(1-ssim(final_result, target).mean())
# loss_final = loss_l1+loss_ssim+loss_per+loss_l1down  
# print("total", ":", loss_final,  "loss_l1", ":", loss_l1,"loss_ssim", ":", loss_ssim)

# 测试UHD模型权重迁移效果

net = InteractNet()
path = "/home/user/data/work/UHDpath/UHD.ckpt"
param_dict = load_checkpoint(path)
load_param_into_net(net, param_dict)
net.set_train(False)

enhance,out_scale_1 = net(final_result)
print(out_scale_1.asnumpy())