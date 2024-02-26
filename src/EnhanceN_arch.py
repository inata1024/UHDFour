"""
## ECCV 2022
"""

# --- Imports --- #
import numpy as np

import mindspore
from mindspore import nn, ops

class UNetConvBlock(nn.Cell):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()

        self.identity = nn.Conv2d(in_size, out_size, 1, 1, has_bias=True)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size//2, affine=True)
        self.use_HIN = use_HIN

        self.out_size = out_size

    def construct(self, x):
        
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = ops.Split(1,2)(out)
            out = ops.concat((self.norm(out_1), out_2), 1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out



class InvBlock(nn.Cell):
    def __init__(self, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = UNetConvBlock(self.split_len2, self.split_len1)
        self.G = UNetConvBlock(self.split_len1, self.split_len2)
        self.H = UNetConvBlock(self.split_len1, self.split_len2)

        # self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def construct(self, x):
        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (ops.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(ops.exp(self.s)) + self.G(y1)  # 2 channel
        out = ops.concat((y1, y2), 1)

        return out



class SpaBlock(nn.Cell):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = InvBlock(nc,nc//2)

    def construct(self, x):
        yy=self.block(x)

        return x+yy



class FreBlock(nn.Cell):
    def __init__(self, nc):
        super(FreBlock, self).__init__()

        self.processmag = nn.SequentialCell([
            nn.Conv2d(nc,nc,1,1,has_bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nc,nc,1,1,has_bias=True)])
        self.processpha = nn.SequentialCell([
            nn.Conv2d(nc, nc, 1, 1, has_bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nc, nc, 1, 1, has_bias=True)])

    def construct(self,x):
        mag = ops.ComplexAbs()(x)
        pha = ops.angle(x)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        real = mag * ops.cos(pha)
        imag = mag * ops.sin(pha)
        x_out = ops.Complex()(real, imag)

        return x_out


class FreBlockAdjust(nn.Cell):
    def __init__(self, nc):
        super(FreBlockAdjust, self).__init__()
        self.processmag = nn.SequentialCell([
            nn.Conv2d(nc,nc,1,1,has_bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nc,nc,1,1,has_bias=True)])
        self.processpha = nn.SequentialCell([
            nn.Conv2d(nc, nc, 1, 1, has_bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nc, nc, 1, 1, has_bias=True)])
        self.sft = SFT(nc)
        self.cat = nn.Conv2d(2*nc,nc,1,1,has_bias=True)

    def construct(self,x, y_amp, y_phase):
        mag = ops.ComplexAbs()(x)
        pha = ops.angle(x)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        mag = self.sft(mag, y_amp)
        pha = self.cat(ops.concat([y_phase,pha],1))
        real = mag * ops.cos(pha)
        imag = mag * ops.sin(pha)
        x_out = ops.Complex()(real, imag)

        return x_out



# def mean_channels(F):
#     assert(F.dim() == 4)
#     spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
#     return spatial_sum / (F.size(2) * F.size(3))


# def stdv_channels(F):
#     assert(F.dim() == 4)
#     F_mean = mean_channels(F)
#     F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
#     return F_variance.pow(0.5)


class ProcessBlock(nn.Cell):
    def __init__(self, in_nc):
        super(ProcessBlock,self).__init__()
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlock(in_nc)

        self.frequency_spatial = nn.Conv2d(in_nc,in_nc,3,1,pad_mode='pad',padding=1,has_bias=True)
        self.spatial_frequency = nn.Conv2d(in_nc,in_nc,3,1,pad_mode='pad',padding=1,has_bias=True)
        self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,has_bias=True)



    def construct(self, x):
        xori = x
        _, _, H, W = x.shape
        rfft2 = ops.FFTWithSize(2, False, True, norm='backward')
        x_freq = rfft2(x)
        x = self.spatial_process(x) # c=192
        x_freq = self.frequency_process(x_freq)
        irfft2 = ops.FFTWithSize(2, True, True, norm='backward',signal_sizes=(H,W))
        x_freq_spatial = irfft2(x_freq)
        
        xcat = ops.concat([x,x_freq_spatial],1)
        x_out = self.cat(xcat)

        return x_out+xori


class ProcessBlockAdjust(nn.Cell):
    def __init__(self, in_nc):
        super(ProcessBlockAdjust,self).__init__()
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlockAdjust(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc,in_nc,3,1,pad_mode='pad',padding=1,has_bias=True)
        self.spatial_frequency = nn.Conv2d(in_nc,in_nc,3,1,pad_mode='pad',padding=1,has_bias=True)
        self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,has_bias=True)


    def construct(self, x, y_amp, y_phase):
        xori = x
        _, _, H, W = x.shape
        rfft2 = ops.FFTWithSize(2, False, True, norm='backward')
        x_freq = rfft2(x)

        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq, y_amp, y_phase)
        irfft2 = ops.FFTWithSize(2, True, True, norm='backward',signal_sizes=(H,W))
        x_freq_spatial = irfft2(x_freq)


        xcat = ops.concat([x,x_freq_spatial],1)
        x_out = self.cat(xcat)

        return x_out+xori


class SFT(nn.Cell):
    def __init__(self, nc):
        super(SFT,self).__init__()
        self.convmul = nn.Conv2d(nc,nc,3,1,pad_mode='pad',padding=1,has_bias=True)
        self.convadd = nn.Conv2d(nc, nc, 3, 1, pad_mode='pad',padding=1,has_bias=True)
        self.convfuse = nn.Conv2d(2*nc, nc, 1, 1, has_bias=True)

    def construct(self, x, res):
        # res = res.detach()
        mul = self.convmul(res)
        add = self.convadd(res)
        fuse = self.convfuse(ops.concat([x,mul*x+add],1))
        return fuse


# def coeff_apply(InputTensor, CoeffTensor, isoffset=True):
#     if not isoffset:
#         raise ValueError("No-offset is not implemented.")
#     bIn, cIn, hIn, wIn = InputTensor.shape
#     bCo, cCo, hCo, wCo = CoeffTensor.shape
#     assert hIn == hCo and wIn == wCo, 'Wrong dimension: In:%dx%d != Co:%dx%d' % (hIn, wIn, hCo, wCo)
#     if isoffset:
#         assert cCo % (cIn + 1) == 0, 'The dimension of Coeff and Input is mismatching with offset.'
#         cOut = cCo / (cIn + 1)
#     else:
#         assert cCo % cIn == 0, 'The dimension of Coeff and Input is mismatching without offset.'
#         cOut = cCo / cIn
#     outList = []

#     if isoffset:
#         for i in range(int(cOut)):
#             Oc = CoeffTensor[:, cIn + (cIn + 1) * i:cIn + (cIn + 1) * i + 1, :, :]
#             Oc = Oc + torch.sum(CoeffTensor[:, (cIn + 1) * i:(cIn + 1) * i + cIn, :, :] * InputTensor,
#                             dim=1, keepdim=True)
#             outList.append(Oc)

#     return torch.cat(outList, dim=1)


class HighNet(nn.Cell):
    def __init__(self, nc):
        super(HighNet,self).__init__()
        self.conv0 = nn.PixelUnshuffle(8)
        self.conv1 = ProcessBlockAdjust(nc*12)
        # self.conv2 = ProcessBlockAdjust(nc)
        self.conv3 = ProcessBlock(nc*12)
        self.conv4 = ProcessBlock(nc*12)
        self.conv5 = nn.PixelShuffle(8)
        self.convout = nn.Conv2d(nc*12//64, 3, 3, 1, pad_mode='pad',padding=1,has_bias=True)
        self.trans = nn.Conv2d(6,16,1,1,has_bias=True)
        self.con_temp1 = nn.Conv2d(16,16,3,1,pad_mode='pad',padding=1,has_bias=True)
        self.con_temp2 = nn.Conv2d(16,16,3,1,pad_mode='pad',padding=1,has_bias=True)
        self.con_temp3 = nn.Conv2d(16,3,3,1,pad_mode='pad',padding=1,has_bias=True)
        self.LeakyReLU=nn.LeakyReLU(0.1)
    def construct(self,x, y_down, y_down_amp, y_down_phase):
        x_ori = x
        x = self.conv0(x) #3*64=192

        x1 = self.conv1(x, y_down_amp, y_down_phase)
        # x2 = self.conv2(x1, y_down_amp, y_down_phase)

        #x3 = self.conv3(x1)
        #x4 = self.conv4(x3)
        x5 = self.conv5(x1)

        xout_temp = self.convout(x5)
        y_aff = self.trans(ops.concat([ops.interpolate(y_down, mode='bilinear',size=(int(y_down.shape[2]*8), int(y_down.shape[3]*8))), xout_temp], 1))
        #con_temp1=self.LeakyReLU(self.con_temp1(y_aff))
        #con_temp2=self.LeakyReLU(self.con_temp2(con_temp1))
        xout=self.con_temp3(y_aff)
        #xout = coeff_apply(x_ori, y_aff)+xout

        return xout

class LowNet(nn.Cell):
    def __init__(self, in_nc=8, nc=192):
        super(LowNet,self).__init__()
        self.conv0 = nn.Conv2d(in_nc,nc,1,1,has_bias=True)
        self.conv1 = ProcessBlock(nc)
        self.downsample1 = nn.Conv2d(nc,nc*2,stride=2,kernel_size=2,pad_mode='valid',has_bias=True)
        self.conv2 = ProcessBlock(nc*2)
        self.downsample2 = nn.Conv2d(nc*2,nc*3,stride=2,kernel_size=2,pad_mode='valid',has_bias=True)
        self.conv3 = ProcessBlock(nc*3)
        self.up1 = nn.Conv2dTranspose(nc*5,nc*2,1,1,has_bias=True)
        self.conv4 = ProcessBlock(nc*2)
        self.up2 = nn.Conv2dTranspose(nc*3,nc*1,1,has_bias=True)
        self.conv5 = ProcessBlock(nc)
        self.convout = nn.Conv2d(nc,nc,1,1,has_bias=True)
        self.convoutfinal = nn.Conv2d(nc, 3, 1, 1, has_bias=True)

        self.transamp = nn.Conv2d(nc,nc,1,1,has_bias=True)
        self.transpha = nn.Conv2d(nc,nc, 1, 1,has_bias=True)

    def construct(self, x):
        x = self.conv0(x)
        x01 = self.conv1(x)
        x1 = self.downsample1(x01)
        x12 = self.conv2(x1)
        x2 = self.downsample2(x12)
        x3 = self.conv3(x2)
        x34 = self.up1(ops.concat([ops.interpolate(x3,size=(x12.shape[2],x12.shape[3]),mode='bilinear'),x12],1))
        x4 = self.conv4(x34)
        x4 = self.up2(ops.concat([ops.interpolate(x4,size=(x01.shape[2],x01.shape[3]),mode='bilinear'),x01],1))
        x5 = self.conv5(x4)
        xout = self.convout(x5)

        rfft2 = ops.FFTWithSize(2, False, True, norm='backward')
        #xout_fre =  torch.fft.rfft2(xout, norm='backward')
        xout_fre =  rfft2(xout)
        xout_fre_amp,xout_fre_phase = ops.ComplexAbs()(xout_fre), ops.angle(xout_fre)
        xfinal = self.convoutfinal(xout)

        return xfinal,self.transamp(xout_fre_amp),self.transpha(xout_fre_phase)



class InteractNet(nn.Cell):
    def __init__(self, nc=16):
        super(InteractNet,self).__init__()
        self.extract =  nn.Conv2d(3,nc//2,1,1,has_bias=True)
        self.lownet = LowNet(nc//2, nc*12)
        self.highnet = HighNet(nc)

    def construct(self, x):
        # print(f'x has nan:{x.isnan().any()}')
        x_f = self.extract(x) # c=8
        # 'For 'interpolate', 'scale_factor' option cannot currently be set with the mode = bilinear and dim = 4D.'
        # replace with 'size' parameter
        x_f_down = ops.interpolate(x_f,mode='bilinear',size=(int(x_f.shape[2]*0.125), int(x_f.shape[3]*0.125)))
        y_down, y_down_amp, y_down_phase = self.lownet(x_f_down)
        y = self.highnet(x,y_down, y_down_amp,y_down_phase)
        
        return y, y_down