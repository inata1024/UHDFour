import mindspore as ms
import torch
from src.EnhanceN_arch import InteractNet

# https://www.mindspore.cn/docs/zh-CN/r2.2/migration_guide/sample_code.html
def pytorch_params(pth_file):
    par_dict = torch.load(pth_file, map_location='cpu')
    pt_params = {}
    _pt_params = {} # 去除前面的modules.
    for name in par_dict:
        parameter = par_dict[name]
        print(name[7:], parameter.numpy().shape)
        _pt_params[name[7:]] = parameter.numpy()
    return _pt_params



# 通过MindSpore的Cell，打印Cell里所有参数的参数名和shape，返回参数字典
def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(name, value.shape)
        ms_params[name] = value
    return ms_params


def param_convert(ms_params, pt_params, ckpt_path):
    # 参数名映射字典
    bn_ms2pt = {"gamma": "weight",
                "beta": "bias",
                "moving_mean": "running_mean",
                "moving_variance": "running_var"}
    new_params_list = []
    for ms_param in ms_params.keys():
        # 在参数列表中，只有包含bn和downsample.1的参数是BatchNorm算子的参数
        if "norm" in ms_param:
            ms_param_item = ms_param.split(".")
            pt_param_item = ms_param_item[:-1] + [bn_ms2pt[ms_param_item[-1]]]
            pt_param = ".".join(pt_param_item)
            # 如找到参数对应且shape一致，加入到参数列表
            if pt_param in pt_params and pt_params[pt_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[pt_param]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
        # 其他参数
        else:
            # 如找到参数对应且shape一致，加入到参数列表
            if ms_param in pt_params and pt_params[ms_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[ms_param]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
    # 保存成MindSpore的checkpoint
    ms.save_checkpoint(new_params_list, ckpt_path)

ckpt_path = "/home/user/data/work/UHDpath/UHD.ckpt"
param_convert(mindspore_params(InteractNet()), pytorch_params('/home/user/data/work/UHDFour_code-main/ckpts/UHD_checkpoint.pt'), ckpt_path)
# mindspore_params(InteractNet())