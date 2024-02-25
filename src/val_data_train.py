# --- Imports --- #
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import imghdr
import random
import numpy as np
import PIL

import mindspore.dataset as ds
from mindspore import dtype as mstype
import mindspore.dataset.vision as vision
from mindspore.dataset.transforms import Compose



# --- Validation/test dataset --- #
class ValGenerator_train:
    def __init__(self, val_data_dir):
        super().__init__()
        val_list = val_data_dir + 'data_list.txt'#'final_test_datalist.txt'eval15/datalist.txt
        with open(val_list) as f:
            contents = f.readlines() 
            lowlight_names = [i.strip() for i in contents]
            gt_names = lowlight_names#[i.split('_')[0] + '.png' for i in lowlight_names]
 
        self.lowlight_names = lowlight_names
        self.gt_names = gt_names 
        self.val_data_dir = val_data_dir
        self.data_list=val_list
    def get_images(self, index):
        lowlight_name = self.lowlight_names[index] 
        gt_name = self.gt_names[index]
        lowlight_img = Image.open(self.val_data_dir + 'input/' + lowlight_name)#eval15/low/
        gt_img = Image.open(self.val_data_dir + 'gt/' + gt_name) #eval15/high/      
        return lowlight_img, gt_img, lowlight_name #

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self): 
        return len(self.lowlight_names)


def make_dataset(path, batch_size=1, shuffle=False):
    """
    Make a dataset for training.
    """
    train_dataset = ds.GeneratorDataset(
        ValGenerator_train(path),
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
