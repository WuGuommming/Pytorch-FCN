import os
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as f
from torch.autograd import Variable
import torchvision.transforms as tfs
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
from FCN import FCNs, FCN8s, FCN16s, FCN32s, VGGNet

print('cuda yes!' if torch.cuda.is_available() else 'no cuda')

# 标签
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']

num_classes = len(classes)
# 标签对应的颜色
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
            [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

cm2lbl = np.zeros(256 ** 3)
# RGB三元组到标签下标的映射
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def image_transforms(data, height, width):
    # data = data.crop([0, 0, 224, 224])
    # 将数据转换成tensor，并且做标准化处理
    im_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data = im_tfs(data)
    return data


all_model = {"FCNs": FCNs,
             "FCN8s": FCN8s,
             "FCN16s": FCN16s,
             "FCN32s": FCN32s,
             }
tar_num = '16s'
tar_model = 'FCN' + tar_num

vgg_model = VGGNet(requires_grad=True)
net = all_model[tar_model](pretrained_net=vgg_model, n_class=num_classes).cuda()
net.load_state_dict(torch.load("F:\\workspace\\model\\fcn_pytorch_" + tar_num + ".pth"))
net.eval()

testRoot = 'F:\\workspace\\test\\'  # 图片
imgRoot = os.listdir(testRoot)

cnt = 0
for oneRoot in imgRoot:
    img = os.path.join(testRoot, oneRoot)
    onedata = Image.open(img)
    onedata = onedata.crop([0, 0, 224, 224])
    onedata.save('F:\\workspace\\crop\\' + oneRoot)
    onedata = image_transforms(onedata, 224, 224)
    onedata = onedata.unsqueeze(0)

    onedata = Variable(onedata).cuda()
    out = net(onedata)
    output = f.log_softmax(out, dim=1)
    print('-----------------')
    cnt += 1
    # print(output.shape) [1, 21, 224, 224]
    print(str(cnt) + " ok " + oneRoot)
    outindx = torch.argmax(output, dim=1).cpu().data.numpy()  # = max(output, dim=10)[1] .max()返回两个对象 一个是值一个是下标
    outimg = [[[0, 0, 0] for i in range(224)] for j in range(224)]

    for i in range(224):
        for j in range(224):
            oneClass = outindx[0][i][j]
            outimg[i][j] = colormap[oneClass]

    outimg = np.array(outimg)
    pre = Image.fromarray(outimg.astype('uint8'), mode='RGB')
    '''
    img = Image.new('RGB', (224, 224))

    for i in range(224):
        for j in range(224):
            img.putpixel((j, i), (outimg[i][j][0], outimg[i][j][1], outimg[i][j][2]))
    '''

    saveRoot = 'F:\\workspace\\res_%s\\' % tar_model
    if not os.path.exists(saveRoot):
        os.makedirs(saveRoot)
    pre.save(saveRoot + oneRoot)
