## Embedding Fourier for Ultra-High-Definition Low-Light Image Enhancement (ICLR 2023 Oral)

This is the MindSpore version of Embedding Fourier for Ultra-High-Definition Low-Light Image Enhancement, ICLR 2023

The official PyTorch implementation, pretrained models and examples are available at https://github.com/Li-Chongyi/UHDFour_code/tree/main

## Requirements

1. python 3.8.18
2. minspore 2.2.11 https://www.mindspore.cn/install/ 
3. cuda 11.1


## Train

```bash
python ./src/train.py
```

Use `python ./lowlight_train.py --help` for more details.

For the perceptual loss used in the paper, you can download the pre-trained VGG19 model from  [Baidu Disk (Key: 1234)](https://pan.baidu.com/s/1gQbONGdvGcf5iDrM5HxbDw).


## Test

```bash
python ./src/test_PSNR.py
```

