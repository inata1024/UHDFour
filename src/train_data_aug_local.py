# train_data_aug_local.py
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


# --- Training dataset --- #
class TrainGenerator:
    def __init__(self, crop_size, train_data_dir):
        super().__init__()

        train_list = train_data_dir +'data_list.txt'  #'datalist.txt''train_list_recap.txt' 'fitered_trainingdata.txt'
        with open(train_list) as f:
            contents = f.readlines()
            lowlight_names = [i.strip() for i in contents]
            gt_names = lowlight_names#[i.split('_')[0] for i in lowlight_names]

        self.lowlight_names = lowlight_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.size_w = crop_size[0]
        self.size_h = crop_size[1]
        self.train_data_dir = train_data_dir

    def get_params(self, img, output_size):
        w, h = img.size
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        return i, j, th, tw
    
    def get_images(self, index):
        lowlight_name = self.lowlight_names[index]
        gt_name = self.gt_names[index]


        lowlight = Image.open(self.train_data_dir + 'input/' + lowlight_name).convert('RGB') #'input_unprocess_aligned/' v
        clear = Image.open(self.train_data_dir + 'gt/' + gt_name ).convert('RGB')  #'gt_unprocess_aligned/''high/'


        if not isinstance(self.crop_size,str):
            # top left height width
            i,j,h,w = self.get_params(lowlight,output_size=(self.size_w,self.size_h))
            # 直接用PIL的方法 left, top, right, bottom
            # 这里top和bottom和字面意思相反 https://blog.csdn.net/weixin_43135178/article/details/126418747
            lowlight=lowlight.crop((j,i,j+w,i+h))
            clear=clear.crop((j,i,j+w,i+h))
    
        data, target=self.augData(lowlight.convert("RGB") ,clear.convert("RGB") )

        return data, target #, lowlight.resize((width/8, height/8)),gt.resize((width/8, height/8))#,factor
    def augData(self,data,target):
        if np.random.rand() < 0.5:
            data = data.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)

        angle = np.random.choice([0, 90, 180, 270])
        data = data.rotate(angle, expand=True)
        target = target.rotate(angle, expand=True)

        return  data, target

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.lowlight_names)



def make_dataset(size, path, batch_size=8, shuffle=True):
    """
    Make a dataset for training.
    """
    train_dataset = ds.GeneratorDataset(
        TrainGenerator(size, path),
        ["data", "target"],
        shuffle = shuffle
    )

    trans = Compose([vision.ToTensor()])

    # 之前的数据处理这里也可以做
    train_dataset = train_dataset.map(operations=trans, input_columns="data")
    train_dataset = train_dataset.map(operations=trans, input_columns="target")

    # buffer_size默认为10
    train_dataset = train_dataset.batch(batch_size)

    return train_dataset



