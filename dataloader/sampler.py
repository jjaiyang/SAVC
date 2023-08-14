import torch
import numpy as np
import copy

'''
__init__方法:
初始化批数量n_batch, 类数量n_cls, 每类样本数量n_per
将所有标签转换为numpy数组
为每个类别建立样本索引m_ind列表
__len__方法:
返回批数量n_batch
__iter__方法:
迭代批数量次
随机采样n_cls个类别
对每个类别随机采样n_per个样本索引
横向堆叠样本索引,并reshape为一维
产生abcdabcd形式的样本索引
迭代yield样本索引batch

所以 CategoriesSampler 实现了数据集的类别均衡采样:
每次迭代按照指定类别数、样本数采样
保证每个batch包含均衡的各类别样本
迭代产生类别均衡的batch样本索引
'''
class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, ):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indexs,e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch
