#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from math import log10
from datetime import datetime
from PIL import Image

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import imageio
import seaborn as sns
import mmcv
import os
from skimage import color
import numpy as np
#import matplotlib.pyplot as plt
import cv2
#import matplotlib
#matplotlib.use('AGG')#或者PDF, SVG或PS
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from PIL import Image
from mindspore import nn, ops

def clear_line():
    """Clears line from any characters."""

    print('\r{}'.format(' ' * 80), end='\r')


def progress_bar(batch_idx, num_batches, report_interval, train_loss):
    """Neat progress bar to track training."""

    dec = int(np.ceil(np.log10(num_batches)))
    bar_size = 21 + dec
    progress = (batch_idx % report_interval) / report_interval
    fill = int(progress * bar_size) + 1
    print('\rBatch {:>{dec}d} [{}{}] Train loss: {:>1.5f}'.format(batch_idx + 1, '=' * fill + '>', ' ' * (bar_size - fill), train_loss, dec=str(dec)), end='')


def time_elapsed_since(start):
    """Computes elapsed time since start."""

    timedelta = datetime.now() - start
    string = str(timedelta)[:-7]
    ms = int(timedelta.total_seconds() * 1000)

    return string, ms


def show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr):
    """Formats validation error stats."""

    clear_line()
    print('Train time: {} | Valid time: {} | Valid loss: {:>1.5f} | Avg PSNR: {:.2f} dB'.format(epoch_time, valid_time, valid_loss, valid_psnr))


def show_on_report(batch_idx, num_batches, loss, elapsed):
    """Formats training stats."""

    clear_line()
    dec = int(np.ceil(np.log10(num_batches)))
    print('Batch {:>{dec}d} / {:d} | Avg loss: {:>1.5f} | Avg train time / batch: {:d} ms'.format(batch_idx + 1, num_batches, loss, int(elapsed), dec=dec))


def plot_per_epoch(ckpt_dir, title, measurements, y_label):
    """Plots stats (train/valid loss, avg PSNR, etc.)."""

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(measurements) + 1), measurements)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Epoch')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()

    fname = '{}.png'.format(title.replace(' ', '-').lower())
    plot_fname = os.path.join(ckpt_dir, fname)
    plt.savefig(plot_fname, dpi=200)
    plt.close()


def psnr(input, target):
    """Computes peak signal-to-noise ratio."""
    
    return 10 * ops.log10(1 / nn.MSELoss()(input, target))


# def create_montage(img_name, noise_type, save_path, source_t, denoised_t, clean_t, show):
#     """Creates montage for easy comparison."""

#     fig, ax = plt.subplots(1, 3, figsize=(9, 3))
#     fig.canvas.set_window_title(img_name.capitalize()[:-4])

#     # Bring tensors to CPU
#     source_t = source_t.cpu().narrow(0, 0, 3)
#     denoised_t = denoised_t.cpu()
#     clean_t = clean_t.cpu()
    
#     source = tvF.to_pil_image(source_t)
#     denoised = tvF.to_pil_image(torch.clamp(denoised_t, 0, 1))
#     clean = tvF.to_pil_image(clean_t)

#     # Build image montage
#     psnr_vals = [psnr(source_t, clean_t), psnr(denoised_t, clean_t)]
#     titles = ['Input: {:.2f} dB'.format(psnr_vals[0]),
#               'Denoised: {:.2f} dB'.format(psnr_vals[1]),
#               'Ground truth']
#     zipped = zip(titles, [source, denoised, clean])
#     for j, (title, img) in enumerate(zipped):
#         ax[j].imshow(img)
#         ax[j].set_title(title)
#         ax[j].axis('off')

#     # Open pop up window, if requested
#     if show > 0:
#         plt.show()

#     # Save to files
#     fname = os.path.splitext(img_name)[0]
#     source.save(os.path.join(save_path, f'{fname}-{noise_type}-noisy.png'))
#     denoised.save(os.path.join(save_path, f'{fname}-{noise_type}-denoised.png'))
#     fig.savefig(os.path.join(save_path, f'{fname}-{noise_type}-montage.png'), bbox_inches='tight')


class AvgMeter(object):
    """Computes and stores the average and current value.
    Useful for tracking averages such as elapsed times, minibatch losses, etc.
    """

    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0.
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def to_psnr(enhance, gt):
    mse = nn.MSELoss(reduction='none')(enhance, gt)
    mse_split = ops.Split()(mse)
    mse_list = [ops.ReduceMean()(ops.Squeeze()(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(enhance, gt):
    enhance_list = ops.Split()(enhance)
    gt_list = ops.Split()(gt)

    enhance_list_np = [ops.Squeeze()(ops.permute(enhance_list[ind],(0, 2, 3, 1))).asnumpy() for ind in range(len(enhance_list))]
    gt_list_np = [ops.Squeeze()(ops.permute(gt_list[ind],(0, 2, 3, 1))).asnumpy() for ind in range(len(enhance_list))]
    ssim_list = [compare_ssim(enhance_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(enhance_list))]

    return ssim_list
                                  
def tensor2image(tensor):
    """
    transfer tensor to numpy
    """
    img = tensor.asnumpy()
    img *= 255
    img = img.clip(0, 255)
    img = img.astype(np.uint8)
    img = img.transpose((1, 2, 0))
    return img

def save_image(enhance, image_name, category):
    enhance_images = ops.split(enhance, 1)
    batch_num = len(enhance_images)
    File_Path = './results/{}_results'.format(category)
    
    if not os.path.exists(File_Path):
        os.makedirs(File_Path) 
    for ind in range(batch_num):
        img = enhance_images[ind].squeeze(0)
        img = tensor2image(img)
        image = Image.fromarray(img)
        image.save('./results/{}_results/{}'.format(category, image_name.asnumpy()[ind][:-3] + 'png'))

             
def validation_PSNR(UHD_Net_reuse, val_data_loader, category, save_tag=False):
    psnr_list = []
    ssim_list = []

    UHD_Net_reuse.set_grad(False)
    for batch_id, val_data in enumerate(val_data_loader):

        lowlight, gt, image_name = val_data
        lowlight = lowlight 
        gt = gt
        
        enhance,out_scale_1 = UHD_Net_reuse(lowlight)       
            
        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(enhance, gt)) 

        # --- Calculate the average SSIM --- # 
        ssim_list.extend(to_ssim_skimage(enhance, gt))

        # --- Save image --- #
        if save_tag:
            save_image(enhance, image_name, category)

    avr_psnr = sum(psnr_list) / len(psnr_list) 
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim#,enhance

def find_image(img_dir):
    filenames = os.listdir(img_dir)
    for i, filename in enumerate(filenames):
        if not filename.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')):
            filenames.pop(i)
    
    return filenames

def generate_filelist(img_dir, valid=False):
    # get filenames list
    filenames = find_image(img_dir)
    if len(filenames) == 0:
        filenames = find_image(os.path.join(img_dir, 'input'))
        if len(filenames) == 0:
            raise(f"No image in directory: '{img_dir}' or '{os.path.join(img_dir, 'input')}'")

    # write filenames
    filelist_name = 'train_list.txt'
    with open(os.path.join(img_dir, filelist_name), 'w') as f:
        for filename in filenames:
            f.write(filename + '\n')