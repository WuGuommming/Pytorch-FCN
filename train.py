from torch import optim
from torch.autograd import Variable
from datetime import datetime
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as f
import torch
import numpy as np

import VOCdata

from FCN import FCNs, FCN8s, FCN16s, FCN32s, VGGNet

BATCH_SIZE = 5
trainData = DataLoader(VOCdata.vocTrain, batch_size=BATCH_SIZE, shuffle=True)
validData = DataLoader(VOCdata.vocTest, batch_size=BATCH_SIZE, shuffle=True)
vocTrain = VOCdata.vocTrain
vocTest = VOCdata.vocTest
num_classes = len(VOCdata.classes)


# 计算混淆矩阵
def _fast_hist(label_true, label_pred, n_class):
    # mask在和label_true相对应的索引的位置上填入true或者false
    # label_true[mask]会把mask中索引为true的元素输出
    mask = (label_true >= 0) & (label_true < n_class)
    '''
    hist二维数组，可以写成hist[label_true][label_pred]的形式
    最后得到的这个数组的意义就是行下标表示的类别预测成列下标类别的数量
    比如hist[0][1]就表示类别为1的像素点被预测成类别为0的数量
    对角线上就是预测正确的像素点个数
    n_class * label_true[mask].astype(int) + label_pred[mask]计算得到的是二维数组元素
    变成一位数组元素的时候的地址取值(每个元素大小为1)，返回的是一个numpy的list，然后
    '''
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


# label_trues 正确的标签值 label_preds 模型输出的标签值 n_class 数据集中的分类数
def label_accuracy_score(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    # 一个batch里面可能有多个数据
    # 通过迭代器将一个个数据进行计算
    for lt, lp in zip(label_trues, label_preds):
        # numpy.ndarray.flatten将numpy对象拉成1维
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    '''
    acc是准确率 = 预测正确的像素点个数/总的像素点个数
    acc_cls是预测的每一类别的准确率(比如第0行是预测的类别为0的准确率)，然后求平均
    iu是召回率Recall，公式上面给出了
    mean_iu就是对iu求了一个平均
    freq是每一类被预测到的频率
    fwavacc是频率乘以召回率，相当与按出现次数加权
    '''
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    # nanmean会自动忽略nan的元素求平均
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


train_loss = []
train_acc = []
train_acc_cls = []
train_mean_iu = []
train_fwavacc = []

eval_loss = []
eval_acc = []
eval_acc_cls = []
eval_mean_iu = []
eval_fwavacc = []

all_model = {"FCNs": FCNs,
             "FCN8s": FCN8s,
             "FCN16s": FCN16s,
             "FCN32s": FCN32s,
             }
tar_num = '32s'
tar_model = 'FCN' + tar_num


def main():
    vgg_model = VGGNet(requires_grad=True)
    net = all_model[tar_model](pretrained_net=vgg_model, n_class=len(VOCdata.classes))
    net = net.cuda()
    criterion = nn.NLLLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    print('-----------------------train-----------------------')

    for epoch in range(30):
        _train_loss = 0
        _train_acc = 0
        _train_acc_cls = 0
        _train_mean_iu = 0
        _train_fwavacc = 0

        prev_time = datetime.now()
        net = net.train()
        for img_data, img_label in trainData:
            if torch.cuda.is_available:
                im = Variable(img_data).cuda()
                label = Variable(img_label).cuda()
            else:
                im = Variable(img_data)
                label = Variable(img_label)

            # 前向传播
            out = net(im)
            out = f.log_softmax(out, dim=1)
            loss = criterion(out, label)

            # 反向传播
            optimizer.zero_grad()  # 梯度初始化为0
            loss.backward()
            optimizer.step()  # 更新
            _train_loss += loss.item()

            # label_pred输出的是21*224*224的向量 对于每一个点都有21个分类的概率
            # 取概率值最大的那个下标作为模型预测的标签 计算评价指标
            label_pred = out.max(dim=1)[1].data.cpu().numpy()
            label_true = label.data.cpu().numpy()

            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
                _train_acc += acc
                _train_acc_cls += acc_cls
                _train_mean_iu += mean_iu
                _train_fwavacc += fwavacc

        # 记录当前轮的数据
        train_loss.append(_train_loss / len(trainData))
        train_acc.append(_train_acc / len(vocTrain))
        train_acc_cls.append(_train_acc_cls)
        train_mean_iu.append(_train_mean_iu / len(vocTrain))
        train_fwavacc.append(_train_fwavacc)

        # 开始评估这轮训练
        net = net.eval()

        _eval_loss = 0
        _eval_acc = 0
        _eval_acc_cls = 0
        _eval_mean_iu = 0
        _eval_fwavacc = 0

        for img_data, img_label in validData:
            if torch.cuda.is_available():
                im = Variable(img_data).cuda()
                label = Variable(img_label).cuda()
            else:
                im = Variable(img_data)
                label = Variable(img_label)

            # forward

            out = net(im)
            out = f.log_softmax(out, dim=1)
            loss = criterion(out, label)
            _eval_loss += loss.item()

            label_pred = out.max(dim=1)[1].data.cpu().numpy()
            label_true = label.data.cpu().numpy()
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
                _eval_acc += acc
                _eval_acc_cls += acc_cls
                _eval_mean_iu += mean_iu
                _eval_fwavacc += fwavacc

        # 记录当前轮的数据
        eval_loss.append(_eval_loss / len(validData))
        eval_acc.append(_eval_acc / len(vocTest))
        eval_acc_cls.append(_eval_acc_cls)
        eval_mean_iu.append(_eval_mean_iu / len(vocTest))
        eval_fwavacc.append(_eval_fwavacc)

        # 打印当前轮训练的结果
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IU: {:.5f}, \
        Valid Loss: {:.5f}, Valid Acc: {:.5f}, Valid Mean IU: {:.5f} '.format(
            epoch, _train_loss / len(trainData), _train_acc / len(vocTrain), _train_mean_iu / len(vocTrain),
               _eval_loss / len(validData), _eval_acc / len(vocTest), _eval_mean_iu / len(vocTest)))
        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        print(epoch_str + time_str)

    torch.save(net.state_dict(), 'F:\\workspace\\model\\fcn_pytorch_' + tar_num + '.pth')


if __name__ == '__main__':
    main()

