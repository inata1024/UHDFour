# $\rm{[MindSpore]}$ $UHDFour$

本项目包含了以下论文的mindspore实现：

> **Embedding Fourier for Ultra-High-Definition Low-Light Image Enhancement**
>
> Chongyi Li, Chun-Le Guo, Man Zhou, Zhexin Liang, Shangchen Zhou, Ruicheng Feng, Chen Change Loy
> 
> The International Conference on Learning Representations (**ICLR**), 2023
> 
[[Paper](https://li-chongyi.github.io/UHDFour/)]



文章官方版本仓库链接: https://github.com/Li-Chongyi/UHDFour_code

## Requirements

1. python 3.8.18
2. minspore 2.2.11 https://www.mindspore.cn/install/ 
3. cuda 11.1


## Train

```bash
python ./src/train.py
```

Use `python ./train.py --help` for more details.

For the perceptual loss used in the paper, you can download the pre-trained VGG19 model from  [Baidu Disk (Key: 1234)](https://pan.baidu.com/s/1gQbONGdvGcf5iDrM5HxbDw).


## Test

```bash
python ./src/test_PSNR.py
```
