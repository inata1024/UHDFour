#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from utils import *
import os
import json 
import networks
import mindspore_ssim

# new import
from mindspore import nn
import mindspore.experimental.optim as optim
from mindspore import context, value_and_grad
from mindspore import save_checkpoint, load_checkpoint
from mindspore.ops import interpolate
from mindspore.nn import WithLossCell, TrainOneStepCell
from networks import vgg19
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import mindspore.ops.operations as P

# VGG test


class FinalLoss(nn.Cell):
    """
    The total loss of the UHDFour
    """
    def __init__(self, dataset_name, VGG):
        super().__init__()

        # Loss function
        if dataset_name == 'UHD':
            self.scale = 0.125
        else:
            self.scale = 0.5

        self.L1 = nn.SmoothL1Loss(reduction='mean')
        self.L2 = nn.MSELoss()
        self.ssim = nn.SSIM()
        self.VGG  = VGG


    def construct(self, final_result, final_result_down, target):

        loss_l1 = 5*self.L1(final_result,target) 
        
        loss_l1down = 0.5*self.L1(final_result_down, interpolate(target,mode='bilinear',size=(int(target.shape[2]*self.scale), int(target.shape[3]*self.scale)))) 
        result_feature = self.VGG(final_result_down)
        target_feature = self.VGG(interpolate(target,mode='bilinear',size=(int(target.shape[2]*self.scale), int(target.shape[3]*self.scale))))
        loss_per = 0.001*self.L2(result_feature, target_feature) 
        loss_ssim=0.002*(1-self.ssim(final_result, target).mean())
        loss_final = loss_l1+loss_ssim+loss_per+loss_l1down  
        print("total", ":", loss_final,  "loss_l1", ":", loss_l1,"loss_ssim", ":", loss_ssim)

        return loss_final
    
    
class MyWithLossCell(nn.Cell):
   def __init__(self, backbone, loss_fn):
       super(MyWithLossCell, self).__init__(auto_prefix=False)
       self._backbone = backbone
       self._loss_fn = loss_fn

   def construct(self, source, target):
       final_result,final_result_down = self._backbone(source)
       return self._loss_fn(final_result, final_result_down, target)

   @property
   def backbone_network(self):
       return self._backbone


class ClipGradients(nn.Cell):
    """
    Clip gradients.

    Args:
        grads (list): List of gradient tuples.
        clip_type (Tensor): The way to clip, 'value' or 'norm'.
        clip_value (Tensor): Specifies how much to clip.

    Returns:
        List, a list of clipped_grad tuples.
    """

    def __init__(self):
        super(ClipGradients, self).__init__()
        self.clip_by_norm = nn.ClipByNorm()
        self.cast = P.Cast()
        self.dtype = P.DType()

    def construct(self,
                  grads,
                  clip_type,
                  clip_value):
        """Defines the gradients clip."""
        if clip_type not in (0, 1):
            return grads
        new_grads = ()
        for grad in grads:
            dt = self.dtype(grad)
            if clip_type == 0:
                t = C.clip_by_value(grad, self.cast(F.tuple_to_array((-clip_value,)), dt),
                                    self.cast(F.tuple_to_array((clip_value,)), dt))
            else:
                t = self.clip_by_norm(grad, self.cast(
                    F.tuple_to_array((clip_value,)), dt))
            t = self.cast(t, dt)
            new_grads = new_grads + (t,)
        return new_grads


class ClipTrainOneStepCell(TrainOneStepCell):
    """
    Encapsulation class of GRU network training.
    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.
    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
        enable_clip_grad (boolean): If True, clip gradients in ClipTrainOneStepCell. Default: True.
    """

    def __init__(self, network, optimizer, clip_type = 0, enable_clip_grad=True, grad_clip_norm=0.1, sens=1.0):
        super(ClipTrainOneStepCell, self).__init__(network, optimizer)
        self.cast = P.Cast()
        self.hyper_map = C.HyperMap()
        self.clip_gradients = ClipGradients()
        self.enable_clip_grad = enable_clip_grad
        self.grad_clip_norm = grad_clip_norm
        self.clip_type =  clip_type
        self.sens = sens

    def construct(self, x, y):
        """Defines the computation performed."""

        weights = self.weights
        loss = self.network(x, y)
        grads = self.grad(self.network, weights=weights)(x, y, self.sens)
        if self.enable_clip_grad:
            grads = self.clip_gradients(
                grads, self.clip_type, self.grad_clip_norm)
        grads = self.grad_reducer(grads)
        succ = self.optimizer(grads)
        return F.depend(loss, succ)

   


class UHDFour(object):
    """Implementation of UHDFour from Li et al. (2023)."""

    def __init__(self, params, trainable):
        """Initializes model."""

        self.p = params
        self.trainable = trainable
        self._compile()

 
    def _compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""

        print(' UHDFour from Li et al. (2023)')

        # Model (3x3=9 channels for Monte Carlo since it uses 3 HDR buffers)
        if self.p.dataset_name == 'UHD':
            from EnhanceN_arch import InteractNet as UHD_Net
        else:
            from EnhanceN_arch_LOL import InteractNet as UHD_Net
        self.model=UHD_Net()
        # Set optimizer and loss, if in training mode
        if self.trainable: 
            self.optim = optim.Adam(self.model.trainable_params(),
                              lr =self.p.learning_rate, 
                              betas=self.p.adam[:2],
                              eps=self.p.adam[2]) 

            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=2, T_mult=2)

            # self.optim = nn.Adam(self.model.trainable_params(),
            #                   learning_rate =self.p.learning_rate, 
            #                   beta1=self.p.adam[0],
            #                   beta2=self.p.adam[1],
            #                   eps=self.p.adam[2]) 


        # CUDA support
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()
        context.set_context(device_target="GPU")


    def _print_params(self):
        """Formats parameters to print when training."""

        print('Training parameters: ')
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        print()

    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        # Create directory for model checkpoints, if nonexistent
        if first:
            ckpt_dir_name = f'{datetime.now():{self.p.dataset_name}-%m%d-%H%M}'
            if self.p.ckpt_overwrite:
                ckpt_dir_name = self.p.dataset_name

            self.ckpt_dir = os.path.join(self.p.ckpt_save_path, ckpt_dir_name)
            if not os.path.isdir(self.p.ckpt_save_path):
                os.mkdir(self.p.ckpt_save_path)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)

        # Save checkpoint dictionary
        if self.p.ckpt_overwrite:
            fname_unet = '{}/UHDFour-{}.pt'.format(self.ckpt_dir, self.p.dataset_name)
        else:
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = '{}/UHDFour-epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1, valid_loss)
        print('Saving checkpoint to: {}\n'.format(fname_unet))
        save_checkpoint(self.model, fname_unet)

        # Save stats to JSON
        fname_dict = '{}/UHDFour-stats.json'.format(self.ckpt_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)



    def load_model(self, ckpt_fname):
        """Loads model from checkpoint file."""

        print('Loading checkpoint from: {}'.format(ckpt_fname))
        load_checkpoint(ckpt_fname, self.model)



    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
        """Tracks and saves starts after each epoch."""
        # Evaluate model on validation set
        print('\rTesting model on validation set... ', end='')
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time, valid_psnr = self.eval(valid_loader)
        show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr)

        # Decrease learning rate if plateau
        self.scheduler.step()

        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)
        self.save_model(epoch, stats, epoch == 0)

        # Plot stats
        if self.p.plot_stats:
            plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], 'L1_Loss')
            plot_per_epoch(self.ckpt_dir, 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')


    def eval(self, valid_loader):
        """Evaluates denoiser on validation set."""

        self.model.set_train(False)

        valid_start = datetime.now() 
        loss_meter = AvgMeter()
        psnr_meter = AvgMeter()

        for batch_idx, (source, target1,lowlight_name) in enumerate(valid_loader):
            final_result,final_result_down = self.model(source)

            # Update loss
            loss = self.L1(final_result, target1)
            loss_meter.update(loss.item())

            # Compute PSRN
            for i in range(1):
                # 网络运行后返回的Tensor 默认均拷贝到CPU 设备
                psnr_meter.update(psnr(final_result[i], target1[i]).item())

        valid_loss = loss_meter.avg
        valid_time = time_elapsed_since(valid_start)[0] 
        psnr_avg = psnr_meter.avg

        return valid_loss, valid_time, psnr_avg



    def train(self, train_loader, valid_loader):  
        """Trains UHDNet on training set.""" 

        if self.p.ckpt_load_path is not None:
            load_checkpoint(self.p.ckpt_load_path, self.model)
            print('The pretrain model is loaded.') 
        self._print_params() 
        num_batches = len(train_loader)
        assert num_batches % self.p.report_interval == 0, 'Report interval must divide total number of batches'

        # Dictionaries of tracked stats
        stats = {'train_loss': [],
                 'valid_loss': [], 
                 'valid_psnr': []}
                 
        # load VGG19 function
        VGG = vgg19()
        vgg_path = "./pre_trained_VGG19_modle/vgg19.ckpt"
        param_dict = load_checkpoint(vgg_path)
        load_param_into_net(VGG, param_dict)
        VGG.set_train(False)

        loss_fn = FinalLoss(self.p.dataset_name, VGG)
        new_with_loss = MyWithLossCell(self.model, loss_fn)
        train_network = ClipTrainOneStepCell(
            new_with_loss, self.optim, clip_type = 0, enable_clip_grad=True, grad_clip_norm=0.1
        )

        train_start = datetime.now()
        for epoch in range(self.p.nb_epochs):
            print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))
        
            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()

            # Minibatch SGD
            for batch_idx, (source, target) in enumerate(train_loader):
                batch_start = datetime.now()
                progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)

                loss_final = train_network(source, target)
                loss_meter.update(loss_final.item())   
                       
                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                    show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
                    train_loss_meter.update(loss_meter.avg)
                    loss_meter.reset() 
                    time_meter.reset() 

            self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader)
            train_loss_meter.reset()

        train_elapsed = time_elapsed_since(train_start)[0]
        print('Training done! Total elapsed time: {}\n'.format(train_elapsed))
 




