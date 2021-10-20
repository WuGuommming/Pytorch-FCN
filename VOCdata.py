import torch
from torch import nn
import torch.nn.functional as f
import torchvision
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

vocRoot = "D:\\data\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012"


# 图片 JPEGImages\\文件名.jpg
# 标签 SegmentationClass\\文件名.png
# 文件名 ImageSets\\Segmentation\\train.txt 或 val.txt
print('cuda yes! ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')


# 返回图片和标签的路径
def read_images(root=vocRoot, train=True):
    fileName = root + '\\ImageSets\\Segmentation\\' + ('train.txt' if train else 'val.txt')
    with open(fileName, 'r') as f:
        images = f.read().split()

    data = [os.path.join(root, 'JPEGImages', i + '.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i + '.png') for i in images]

    return data, label


# 图片和标签要像素对应 不能缩放 要裁剪

# 标签
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']

# 标签对应的颜色
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]

cm2lbl = np.zeros(256 ** 3)
# RGB三元组到标签下标的映射
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


# 输入图像 输出所有位置对应标签的下标
def image2label(im):
    data = np.array(im, dtype="int32")
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype="int64")


def image_transforms(data, label, height, width):
    data = data.crop([0, 0, height, width])
    label = label.crop([0, 0, height, width])
    # 将数据转换成tensor，并且做标准化处理
    im_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data = im_tfs(data)
    label = image2label(label)
    label = torch.from_numpy(label)
    return data, label


# 定义数据集
class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, train, height, width, transforms):
        self.height = height
        self.width = width
        self.fnum = 0  # 用来记录被过滤的图片数
        self.transforms = transforms
        data_list, label_list = read_images(train=train)
        self.data_list = self.filter(data_list)
        self.label_list = self.filter(label_list)
        if train:
            print("train: load " + str(len(self.data_list)) + " img and label" + ", filter" + str(self.fnum))
        else:
            print("test: load " + str(len(self.data_list)) + " img and label" + ", filter" + str(self.fnum))

    # 过滤长宽不够的图片
    def filter(self, images):
        res = []
        for oneImg in images:
            if (Image.open(oneImg).size[1] >= self.height and
                    Image.open(oneImg).size[0] >= self.width):
                res.append(oneImg)
            else:
                self.fnum = self.fnum + 1
        return res

    # 重载getitem函数，使类可以迭代
    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.transforms(img, label, self.height, self.width)
        return img, label

    def __len__(self):
        return len(self.data_list)


height = 224
width = 224
vocTrain = VOCSegDataset(True, height, width, image_transforms)
vocTest = VOCSegDataset(False, height, width, image_transforms)

