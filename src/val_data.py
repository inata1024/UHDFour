# --- Imports --- #
from PIL import Image
import numpy as np
import os

import mindspore.dataset as ds
from mindspore import dtype as mstype
import mindspore.dataset.vision as vision
from mindspore.dataset.transforms import Compose

# --- Validation/test dataset --- #
class ValGenerator:
    def __init__(self, dataset_name,val_data_dir):
        super().__init__() 
        self.dataset_name = dataset_name
        val_list = os.path.join(val_data_dir, 'data_list.txt')
        with open(val_list) as f:
            contents = f.readlines()
            lowlight_names = [i.strip() for i in contents]
            if self.dataset_name=='UHD' or self.dataset_name=='LOLv1' or self.dataset_name=='LOLv2':
                gt_names = lowlight_names #
            else:
                gt_names = None 
                print('The dataset is not included in this work.')  
        self.lowlight_names = lowlight_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir
        self.data_list=val_list
    def get_images(self, index):
        lowlight_name = self.lowlight_names[index]
        padding = 8
        # build the folder of validation/test data in our way
        if os.path.exists(os.path.join(self.val_data_dir, 'input')):
            lowlight_img = Image.open(os.path.join(self.val_data_dir, 'input', lowlight_name))
            if os.path.exists(os.path.join(self.val_data_dir, 'gt')) :
                gt_name = self.gt_names[index]
                gt_img = Image.open(os.path.join(self.val_data_dir, 'gt', gt_name)) ##   
                a = lowlight_img.size

                a_0 =a[1] - np.mod(a[1],padding)
                a_1 =a[0] - np.mod(a[0],padding)            
                lowlight_crop_img = lowlight_img.crop((0, 0, 0 + a_1, 0+a_0))
                gt_crop_img = gt_img.crop((0, 0, 0 + a_1, 0+a_0))
            else: 
                # the inputs is used to calculate PSNR.
                a = lowlight_img.size
                a_0 =a[1] - np.mod(a[1],padding)
                a_1 =a[0] - np.mod(a[0],padding)            
                lowlight_crop_img = lowlight_img.crop((0, 0, 0 + a_1, 0+a_0))
                gt_crop_img = lowlight_crop_img
        # Any folder containing validation/test images
        else:
            lowlight_img = Image.open(os.path.join(self.val_data_dir, lowlight_name))
            a = lowlight_img.size
            a_0 =a[1] - np.mod(a[1],padding)
            a_1 =a[0] - np.mod(a[0],padding)            
            lowlight_crop_img = lowlight_img.crop((0, 0, 0 + a_1, 0+a_0))
            gt_crop_img = lowlight_crop_img
         
        return lowlight_crop_img, gt_crop_img, lowlight_name


    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.lowlight_names)

def make_dataset(name, path, batch_size=1, shuffle=False):
    """
    Make a dataset for training.
    """
    train_dataset = ds.GeneratorDataset(
        ValGenerator(name, path),
        ["lowlight", "gt", "lowlight_name"]
    )

    trans = Compose([vision.ToTensor()])

    # 之前的数据处理这里也可以做
    train_dataset = train_dataset.map(operations=trans, input_columns="lowlight")
    train_dataset = train_dataset.map(operations=trans, input_columns="gt")

    # buffer_size默认为10
    if shuffle:
      train_dataset = train_dataset.shuffle(buffer_size=10)
    train_dataset = train_dataset.batch(batch_size)

    return train_dataset
