import math
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common import initializer as init
from mindspore.common.initializer import initializer
import argparse



class vgg19(nn.Cell):
    def __init__(self, feature_mode=True, batch_norm=False, num_classes=1000):
        super(vgg19, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        # self.init_weights = init_weights
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm
        self.num_clases = num_classes
        self.features = self.make_layers(self.cfg, batch_norm)
        self.classifier = nn.SequentialCell(
            nn.Dense(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(4096, num_classes),
        )


    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
                else:
                    layers += [conv2d, nn.ReLU()]
                in_channels = v
        return nn.SequentialCell(*layers)

    def construct(self, x):
        if self.feature_mode:
            module_list = list(self.features.cells())
            # print(module_list)
            for l in module_list[0:26]:                 # conv4_4
                x = l(x)
        # if not self.feature_mode:
        #     x = x.view(x.size(0), -1)
        #     x = self.classifier(x)

        return x