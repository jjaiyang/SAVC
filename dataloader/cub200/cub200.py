import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import *

# 已读
'''
修改父类DataLoader，执行augmentation
'''
class CUB200(Dataset):

    def __init__(self, root='./', train=True, index_path=None, index=None,
                 base_sess=None, crop_transform=None, secondary_transform=None):
        self.root = os.path.expanduser(root) # 获取数据集根目录路径
        self.train = train  # training set or test set
        self._pre_operate(self.root)
        self.transform = None
        self.multi_train = False  # training set or test set 判断是否为多尺度训练（multicrop）
        self.crop_transform = crop_transform # 存储裁剪变换方法
        self.secondary_transform = secondary_transform # 存储辅助变换方法
        if isinstance(secondary_transform, list):
            assert (len(secondary_transform) == self.crop_transform.N_large + self.crop_transform.N_small) # 校验secondary_transform列表长度

        if train:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            # '''
            # 这个Normalize所使用的mean和std的值, 通常是在大规模数据集上进行统计得到的, 如ImageNet数据集。这个代码中使用的mean和std的值, 正是在ImageNet数据集上预计算得到的均值和标准差。
            # 之所以使用这些特定的值进行归一化, 主要有以下原因:
            # 把不同图像调整到同一量化范围, 便于网络处理
            #     ImageNet数据集包含各种场景的图片, 每个像素值分布范围不同。使用数据集统计值进行归一化, 可以把所有图片调整到一个标准的量化范围内, 便于网络处理。
            # 减少不同图像之间的光照、对比度等统计差异
            #     不同图片由于拍摄条件等原因, 整体颜色、对比度统计值会有较大差异。使用固定值归一化可以减少这种统计差异, 使网络更专注于学习区分不同特征。
            # 一致性
            #     使用固定的值(如ImageNet统计值), 可以使不同的训练更具可比性, 也有利于使用预训练模型。
            # 更易学习目标任务
            #     减少输入数据的非关联统计变化, 可以帮助网络更加聚焦在学习如何解决目标任务上。
            # '''
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index) # 基础训练从需要的类中取
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path) # 从txt文件中取
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
    # '''
    # text_read:
    # 读取文本文件, 返回文本行列表每行末尾的\n会被strip删除
    # list2dict:
    # 将文本行列表转换为字典通过split分割每行, 第一个元素作为key, 第二个元素作为value如果key重复则报错最终返回key - value的字典它们的作用是把一个id和文本的映射文件, 转换成一个字典方便查找。
    # 例如原文件内容为:
    # 1 flower
    # 2 cat
    # 3 dog
    # text_read后返回['1 flower', '2 cat', '3 dog']
    # list2dict后返回:
    # {1: 'flower', 2: 'cat', 3: 'dog'}
    # '''
    def text_read(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self, list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def _pre_operate(self, root): # 预处理路径等加载对应的图片样本
        image_file = os.path.join(root, 'CUB_200_2011/images.txt')
        split_file = os.path.join(root, 'CUB_200_2011/train_test_split.txt')
        class_file = os.path.join(root, 'CUB_200_2011/image_class_labels.txt')
        id2image = self.list2dict(self.text_read(image_file)) # 读取图像文件名文件，转换为id到图像名字典，id作为字典的键,名称作为字典的值
        id2train = self.list2dict(self.text_read(split_file)) # 1: train images; 0: test iamges
        id2class = self.list2dict(self.text_read(class_file))
        train_idx = []
        test_idx = []
        for k in sorted(id2train.keys()): # 遍历分割字典的id
            if id2train[k] == '1':
                train_idx.append(k)
            else:
                test_idx.append(k)

        self.data = []
        self.targets = []
        self.data2label = {}
        if self.train: # 如果是train，则保存对应训练的图片路径到data，对应的label到tragets，两相对应的路径类别对到字典data2label
            for k in train_idx: # 所有的train图像路径
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

        else:
            for k in test_idx: # 所有的test图像路径
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

    def SelectfromTxt(self, data2label, index_path):
        index = open(index_path).read().splitlines() # 打开索引文件,读取内容并分割成行存入index列表。 index_path应该是一个包含有后缀的图像文件名的文件
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.root, i) # 拼接全路径。
            data_tmp.append(img_path) # 添加至图像路径列表。
            targets_tmp.append(data2label[img_path]) # 从字典中取出该图像文件路径对应的标签,添加至标签列表。

        return data_tmp, targets_tmp
    # '''
    # 假设原始的数据集有4个样本, 包括:
    # data = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']
    # targets = [0, 1, 1, 2]
    # 这表示有3个类别, img1为类别0, img2和img3为类别1, img4为类别2。
    # 现在我们筛选索引 index 为[1, 2], 表示需要类别1和类别2的样本。
    # 执行过程如下:
    # i = 1, 在targets中找到类别标签等于1的样本索引为[1, 2]。
    # 遍历[1, 2], 将data[1], data[2]和targets[1], targets[2]添加到tmp中。
    # i = 2, 找到类别标签等于2的样本索引为[3]。 遍历[3], 添加data[3]和targets[3]到tmp中。
    # 结果:
    # data_tmp = ['img2.jpg', 'img3.jpg', 'img4.jpg']
    # targets_tmp = [1, 1, 2]
    # '''
    def SelectfromClasses(self, data, targets, index): # 通过类别索引index([0 1 2 3 4 5 6]),从原始数据集中筛选出特定类别的样本
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0] # 找到targets中类别标签等于i的样本索引。
            for j in ind_cl: # 遍历该类别索引。
                data_tmp.append(data[j]) # 添加对应的图像路径。
                targets_tmp.append(targets[j]) # 添加对应的标签。

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i): # DataLoader可以按索引取图像样本
        path, targets = self.data[i], self.targets[i] # 从data和targets列表中取出第i个样本的图像路径和标签。
        if self.multi_train: # 判断是否为多尺度训练模式。
            image = Image.open(path).convert('RGB') # 打开图像路径对应的图片,转换为RGB格式。
            classify_image = [self.transform(image)] # 对图片应用transform转换,得到分类网络的输入图像。
            multi_crop, multi_crop_params = self.crop_transform(image) # 对图片应用crop_transform,生成多尺度裁剪样本
            assert (len(multi_crop) == self.crop_transform.N_large + self.crop_transform.N_small) # 断言生成的裁剪样本数等于预定义的大尺度和小尺度裁剪数之和。
            if isinstance(self.secondary_transform, list): # 判断secondary_transform是否为列表形式。
                multi_crop = [tf(x) for tf, x in zip(self.secondary_transform, multi_crop)] # 如果是列表,则对每个裁剪样本分别应用对应的secondary transform。
            else: # 如果不是列表形式。
                multi_crop = [self.secondary_transform(x) for x in multi_crop] # 则对所有裁剪样本统一应用同一个secondary transform。
            total_image = classify_image + multi_crop # 拼接分类网络输入图片和裁剪增强图片,得到最终输入样本。之所以要构建成一个列表,主要是为了同时送入分类网络和对比学习中
        else: # 如果不是多尺度训练模式。
            total_image = self.transform(Image.open(path).convert('RGB')) # 直接对图片应用transform,得到网络输入样本。
        return total_image, targets # 返回图片样本和标签。
