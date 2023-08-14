import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.resnet18_encoder import *
from models.resnet20_cifar import *

'''
加载clip和swin-T
'''
import clip
import timm

import os
import re
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


'''
定义了Backbone网络MYNET,包含resnet18和resnet20作为编码器。
在__init__中构建网络,包含query编码器,key编码器,全连接层,queue等。
定义了momentum update key encoder的方法_momentum_update_key_encoder,用于更新key encoder。
定义了dequeue和enqueue的方法_dequeue_and_enqueue,用于更新queue。
定义了query和key的编码方法encode_q和encode_k。
    在forward中实现了训练和测试时的前向传播过程。
    在训练时计算classify logits,global contrastive logits,small contrastive logits。
    计算global和small的positive logits和negative logits。
    根据queue更新target。
    在测试时进行nearest class mean预测。
定义了更新fc层的方法update_fc,用于类增量学习。
定义了利用平均特征更新fc的方法update_fc_avg。
定义了计算logits的方法get_logits。
'''

'''
cub200时,网络参数:
编码器backbone: resnet18
输入:224x224x3
conv1: 7x7, 64 filters, stride 2, 输出 112x112x64
maxpool1: 3x3, stride 2, 输出 56x56x64
layer1:
    2个basic block
    [3x3, 64] - [3x3, 64]
    输出 56x56x64
layer2:
    2个basic block
    [3x3, 128] - [3x3, 128]
    通过1x1 conv下采样,输出 28x28x128
layer3:
    2个basic block
    [3x3, 256] - [3x3, 256]
    通过1x1 conv下采样,输出 14x14x256
layer4:
    2个basic block
    [3x3, 512] - [3x3, 512]
    通过1x1 conv下采样,输出 7x7x512
avg pool: 7x7 -> 1x1
全连接层fc:
    输入512个特征
    输出100个类别得分
'''
# 定义文本编码器，调用预训练的CLIP
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 不需要训练,冻结参数
        self.model, self.preprocess = clip.load("ViT-B/32", device='cuda' if torch.cuda.is_available() else "cpu",
                                                jit=False)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, texts):
        texts = clip.tokenize(texts).to(device='cuda' if torch.cuda.is_available() else "cpu")
        text_features = self.model.encode_text(texts)

        return text_features

class Text2Patches(nn.Module):

    def __init__(self):
        super().__init__()
        self.mapping = nn.Linear(512, 224 * 224)

    def forward(self, text_features):
        # (batch_size, 512) -> (batch_size, 3, 512)
        text_features = text_features.unsqueeze(1).repeat(1, 3, 1)

        # (batch_size, 3, 512) -> (batch_size, 3, 50176)
        patches = self.mapping(text_features)

        # (batch_size, 3, 50176) -> (batch_size, 3, 224, 224)
        patches = patches.view(patches.size(0), patches.size(1), 224, 224)

        return patches

class MYNET(nn.Module):

    def __init__(self, args, mode=None, trans=1):
        super().__init__()
        self.mode = mode
        self.args = args

        # 初始化CLIP文本编码器等等
        self.text_encoder = TextEncoder().cuda()
        # self.image_encoder = timm.create_model('swin_tiny_patch4_window7_224.ms_in22k_ft_in1k', pretrained=True, num_classes=0)
        self.text_to_patches = Text2Patches().cuda()

        if self.args.dataset in ['cifar100']:
            self.encoder_q = resnet20(num_classes=self.args.moco_dim)
            self.encoder_k = resnet20(num_classes=self.args.moco_dim)
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet']:
            self.encoder_q = resnet18(False, args, num_classes=self.args.moco_dim)  # pretrained=False
            self.encoder_k = resnet18(False, args, num_classes=self.args.moco_dim)  # pretrained=False
            self.num_features = 512
        if self.args.dataset == 'cub200':
            # self.encoder_q = resnet18(True, args, num_classes=self.args.moco_dim)
            self.encoder_q = timm.create_model('swin_tiny_patch4_window7_224.ms_in22k_ft_in1k', pretrained=True, num_classes=self.args.moco_dim).cuda()
            self.encoder_k = timm.create_model('swin_tiny_patch4_window7_224.ms_in22k_ft_in1k', pretrained=True, num_classes=self.args.moco_dim).cuda()# pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            # self.encoder_k = resnet18(True, args, num_classes=self.args.moco_dim)# pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 768
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        '''
        在SAVC中直接训练SV-T，以上方法就可以了
        如果是加载预训练的backbone的话，需要再添加一层FC层
        '''

        self.fc = nn.Linear(self.num_features, self.args.num_classes*trans, bias=False)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.num_features, 3 * self.num_features, bias=False),
        #     nn.Linear(3 * self.num_features, self.num_features, bias=False),
        #     nn.Linear(self.num_features, self.args.num_classes * trans, bias=False)
        # )

        self.K = self.args.moco_k
        self.m = self.args.moco_m
        self.T = self.args.moco_t
        
        if self.args.mlp:  # hack: brute-force replacement
            self.encoder_q.head.fc = nn.Sequential(
                nn.Linear(self.num_features, self.num_features * 4), nn.ReLU(),
                nn.Linear(self.num_features * 4, self.num_features), self.encoder_q.head.fc
            )
            self.encoder_k.head.fc = nn.Sequential(
                nn.Linear(self.num_features, self.num_features * 4), nn.ReLU(),
                nn.Linear(self.num_features * 4, self.num_features), self.encoder_k.head.fc
            )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        # create the queue
        self.register_buffer("queue", torch.randn(self.args.moco_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("label_queue", torch.zeros(self.K).long() - 1)


    @torch.no_grad()
    def _momentum_update_key_encoder(self, base_sess):
        """
        Momentum update of the key encoder
        """
        if base_sess:
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        else:
            for k, v in self.encoder_q.named_parameters():
                # if k.startswith('fc') or k.startswith('layer4') or k.startswith('layer3'):
                if k.startswith('head.fc') or k.startswith('layers.3'): # 冻结了stage0~2
                    self.encoder_k.state_dict()[k].data = self.encoder_k.state_dict()[k].data * self.m + v.data * (1. - self.m)
                    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
 
        # replace the keys and labels at ptr (dequeue and enqueue)
        if ptr + batch_size > self.K:
            remains = ptr + batch_size - self.K
            self.queue[:, ptr:] = keys.T[:, :batch_size - remains]
            self.queue[:, :remains] = keys.T[:, batch_size - remains:]
            self.label_queue[ptr:] = labels[ :batch_size - remains]
            self.label_queue[ :remains] = labels[batch_size - remains:]
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T # this queue is feature queue
            self.label_queue[ptr:ptr + batch_size] = labels        
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr
    
    def forward_metric(self, x):# 这里输出的是logits
        x, _ = self.encode_q(x) # 这里的前向是图像经过encoder后又池化一次
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)

        return x # joint, contrastive

    '''
    F.adaptive_avg_pool2d会将特征图x适配到1x1的尺寸,即Global Average Pooling。
    这就使得不管输入图像的原始大小（大图和小图）是什么,
    输出的x都被Pool到同样的1x1尺寸,相当于一个固定长度的特征向量。
    所以同一个encoder可以接受不同尺寸的输入图像。
    '''
    def encode_q(self, x):
        a = self.encoder_q.forward_features(x).permute(0, 3, 1, 2) # torch.Size([1, 7, 7, 768]) 转成torch.Size([1, 768, 7, 7])
        b = self.encoder_q(x)
        a = F.adaptive_avg_pool2d(a, 1)
        a = a.squeeze(-1).squeeze(-1)
        return a, b
    
    def encode_k(self, x):
        a = self.encoder_k.forward_features(x).permute(0, 3, 1, 2) # torch.Size([1, 7, 7, 768]) 转成torch.Size([1, 768, 7, 7])
        b = self.encoder_k(x)
        a = F.adaptive_avg_pool2d(a, 1)
        a = a.squeeze(-1).squeeze(-1)
        return a, b # torch.Size([n, 768, 7, 7]) torch.Size([n, 128])

    '''
    im_cla是不经过变换的原始图像,用于分类预测,计算classify logits。
    im_q是对im_cla进行增强和变换得到的查询图像,用于计算contrastive学习中的query特征。
    如果有im_q_small,则是对im_q进行crop得到的小尺寸查询图像,用于计算small contrastive loss。
    im_k是对im_cla进行增强和变换得到的关键图像,用于contrastive学习中的key特征。
    '''
    def forward(self, im_cla, im_q=None, im_k=None, labels=None, im_q_small=None, base_sess=True, txt=None,
                last_epochs_new=False):
        if self.mode != ('encoder' or 'semantic'):
            if im_q == None: # 没有im_q时
                x = self.forward_metric(im_cla) # 仅进行classify logits计算
                return x
            else:
                b = im_q.shape[0] #get batch size
                logits_classify = self.forward_metric(im_cla) # 计算classify logits logits_classify是分类分支输出的logits,会经过softmax和交叉熵损失,进行分类任务的监督训练。
                _, q = self.encode_q(im_q) # 获得query特征q
                q = nn.functional.normalize(q, dim=1) # 对q进行normalize，因为后面直接拿去supcontrastive loss中算内积了
                #看看选哪样的q合适
                feat_dim = q.shape[-1] # 图像向量维度
                q = q.unsqueeze(1) # bs x 1 x dim

                if im_q_small is not None:
                    _, q_small = self.encode_q(im_q_small) # 计算small query特征q_small
                    q_small = q_small.view(b, -1, feat_dim)  # bs x 4 x dim 调整shape 可能是裁剪出了四个small crop图像（num_crops）
                    q_small = nn.functional.normalize(q_small, dim=-1)

                # compute key features
                with torch.no_grad():  # no gradient to keys
                    self._momentum_update_key_encoder(base_sess)  # update the key encoder
                    _, k = self.encode_k(im_k)  # keys: bs x dim
                    k = nn.functional.normalize(k, dim=1)

                # compute logits
                # Einstein sum is more intuitive
                # positive logits: Nx1
                q_global = q
                l_pos_global = (q_global * k.unsqueeze(1)).sum(2).view(-1, 1) # 计算q_global和k的点积,再求和压缩为一个数,view成[batch_size, 1]的SHAPE,得到全局图像的positive logits l_pos_global
                l_pos_small = (q_small * k.unsqueeze(1)).sum(2).view(-1, 1)

                # negative logits: NxK
                l_neg_global = torch.einsum('nc,ck->nk', [q_global.view(-1, feat_dim), self.queue.clone().detach()]) # self.queue是特征队列,SHAPE是[dim, queue_size],包含历史batch的编码
                l_neg_small = torch.einsum('nc,ck->nk', [q_small.view(-1, feat_dim), self.queue.clone().detach()]) # 用einsum计算q_global和self.queue的点积,得到[batch_size, queue_size]的negative logits l_neg_global

                # logits: Nx(1+K)
                logits_global = torch.cat([l_pos_global, l_neg_global], dim=1) # logits_global是全局图像的对比学习logits,SHAPE是[batch_size, 1 + queue_size
                logits_small = torch.cat([l_pos_small, l_neg_small], dim=1)

                # apply temperature
                logits_global /= self.T
                logits_small /= self.T

                '''
                targets的生成过程:
                初始化一个全1的矩阵positive_target,shape是[batch_size, 1],表示每个样本自己是positive sample。
                根据当前batch的labels和存储在queue中的历史labels,生成一个[batch_size, queue_size]的0-1矩阵targets。
                具体是对每个样本:
                    如果该样本的label在queue中存在,则对应的queue位置为1,否则为0。
                    这样就标记出了当前batch与queue中哪些样本是同类的。
                将positive_target和targets在dim=1上拼接,得到targets_global,shape是[batch_size, 1+queue_size]。
                这样targets的第一列是positive sample,其余列标记queue中哪些样本是同类别。
                将targets_global复制多份(copies等于小样本数量),得到targets_small。
                举个详细例子:
                queue = [2, 3, 5, 1, 4]
                batch_labels = [1, 2, 5]
                positive_target = [[1], [1], [1]]
                targets = [[0, 0, 0, 1, 0], # 样本1与queue中的label 1同类 [1, 0, 0, 0, 0], # 样本2与queue中的label 2同类
                [0, 0, 1, 0, 0]] # 样本3与queue中的label 5同类
                targets_global = [[1, 0, 0, 1, 0], [1, 1, 0, 0, 0], [1, 0, 1, 0, 0]] # 在dim=1上拼接positive target
                targets_small = targets_global 重复copy num.crop次
                '''
                # one-hot target from augmented image
                positive_target = torch.ones((b, 1)).cuda() # positive_target是一个全1的tensor,shape是[batch_size, 1],表示query图像自己是positive sample。
                # find same label images from label queue
                # for the query with -1, all 
                # targets通过比较labels和队列queue中的标签,找到与query标签相同的样本,标记为1
                targets = ((labels[:, None] == self.label_queue[None, :]) & (labels[:, None] != -1)).float().cuda()
                targets_global = torch.cat([positive_target, targets], dim=1) # targets_global将positive_target和targets在dim=1上拼接,得到全局图像的targets。
                targets_small = targets_global.repeat_interleave(repeats=self.args.num_crops[1], dim=0) # targets_small重复targets_global得到小样本的targets（因为小图像的类别和之前的也是一样的）
                labels_small = labels.repeat_interleave(repeats=self.args.num_crops[1], dim=0) # labels_small重复labels得到小样本对应的labels。

                # dequeue and enqueue
                if base_sess or (not base_sess and last_epochs_new):
                    self._dequeue_and_enqueue(k, labels)
                
                return logits_classify, logits_global, logits_small, targets_global, targets_small

        '''
        logits_classify: 分类分支输出的logits,通过原始图像预测类别,用于分类任务。计算交叉熵损失。
        logits_global: 对比学习分支中,基于全局query图像和keys计算出的对比logits。
        logits_small: 对比学习分支中,基于小尺寸query图像和keys计算出的对比logits。
        targets_global: 对比学习分支中,基于全局query图像生成的targets,与logits_global一起计算对比loss。
        targets_small: 对比学习分支中,基于小尺寸query图像生成的targets,与logits_small一起计算对比loss。
        '''
        elif self.mode == 'semantic':
            if im_q == None:  # 没有im_q时
                x = self.forward_metric(im_cla)  # 仅进行classify logits计算
                return x
            elif txt == None: # 如果还有其它只用到img_ft = q而不用txt_fc的情况的话加在此处
                b = im_q.shape[0]  # get batch size
                logits_classify = self.forward_metric(im_cla)  # 计算classify logits logits_classify是分类分支输出的logits,会经过softmax和交叉熵损失,进行分类任务的监督训练。
                _, q = self.encode_q(im_q)  # 获得query特征q

                q = nn.functional.normalize(q, dim=1)  # 对q进行normalize
                feat_dim = q.shape[-1]  # 图像向量维度
                q = q.unsqueeze(1)  # bs x 1 x
# 只加到这里之前，txt
                if im_q_small is not None:
                    _, q_small = self.encode_q(im_q_small)  # 计算small query特征q_small
                    q_small = q_small.view(b, -1, feat_dim)  # bs x 4 x dim 调整shape 可能是裁剪出了四个small crop图像（num_crops）
                    q_small = nn.functional.normalize(q_small, dim=-1)

                # compute key features
                with torch.no_grad():  # no gradient to keys
                    self._momentum_update_key_encoder(base_sess)  # update the key encoder
                    _, k = self.encode_k(im_k)  # keys: bs x dim
                    k = nn.functional.normalize(k, dim=1)

                # compute logits
                # Einstein sum is more intuitive
                # positive logits: Nx1
                q_global = q
                l_pos_global = (q_global * k.unsqueeze(1)).sum(2).view(-1,1)  # 计算q_global和k的点积,再求和压缩为一个数,view成[batch_size, 1]的SHAPE,得到全局图像的positive logits l_pos_global
                l_pos_small = (q_small * k.unsqueeze(1)).sum(2).view(-1, 1)

                # negative logits: NxK
                l_neg_global = torch.einsum('nc,ck->nk', [q_global.view(-1, feat_dim),
                                                          self.queue.clone().detach()])  # self.queue是特征队列,SHAPE是[dim, queue_size],包含历史batch的编码
                l_neg_small = torch.einsum('nc,ck->nk', [q_small.view(-1, feat_dim),
                                                         self.queue.clone().detach()])  # 用einsum计算q_global和self.queue的点积,得到[batch_size, queue_size]的negative logits l_neg_global

                # logits: Nx(1+K)
                logits_global = torch.cat([l_pos_global, l_neg_global],
                                          dim=1)  # logits_global是全局图像的对比学习logits,SHAPE是[batch_size, 1 + queue_size
                logits_small = torch.cat([l_pos_small, l_neg_small], dim=1)

                # apply temperature
                logits_global /= self.T
                logits_small /= self.T

                # one-hot target from augmented image
                positive_target = torch.ones((b, 1)).cuda()  # positive_target是一个全1的tensor,shape是[batch_size, 1],表示query图像自己是positive sample。
                # find same label images from label queue
                # for the query with -1, all
                # targets通过比较labels和队列queue中的标签,找到与query标签相同的样本,标记为1
                targets = ((labels[:, None] == self.label_queue[None, :]) & (labels[:, None] != -1)).float().cuda()
                targets_global = torch.cat([positive_target, targets],
                                           dim=1)  # targets_global将positive_target和targets在dim=1上拼接,得到全局图像的targets。
                targets_small = targets_global.repeat_interleave(repeats=self.args.num_crops[1],
                                                                 dim=0)  # targets_small重复targets_global得到小样本的targets（因为小图像的类别和之前的也是一样的）
                labels_small = labels.repeat_interleave(repeats=self.args.num_crops[1], dim=0)  # labels_small重复labels得到小样本对应的labels。

                # dequeue and enqueue
                if base_sess or (not base_sess and last_epochs_new):
                    self._dequeue_and_enqueue(k, labels)

                return logits_classify, logits_global, logits_small, targets_global, targets_small
            else:
                b = im_q.shape[0]  # get batch size
                logits_classify = self.forward_metric(im_cla)  # 计算classify logits logits_classify是分类分支输出的logits,会经过softmax和交叉熵损失,进行分类任务的监督训练。
                _, q = self.encode_q(im_q)  # 获得query特征q

                text_features = self.text_encoder(txt)
                patches = self.text_to_patches(text_features)
                txt_ft = self.encode_q(patches)

                q = nn.functional.normalize(q, dim=1)  # 对q进行normalize
                feat_dim = q.shape[-1]  # 图像向量维度
                q = q.unsqueeze(1)  # bs x 1 x

# 只加到这里之前，txt

                if im_q_small is not None:
                    _, q_small = self.encode_q(im_q_small)  # 计算small query特征q_small
                    q_small = q_small.view(b, -1, feat_dim)  # bs x 4 x dim 调整shape 可能是裁剪出了四个small crop图像（num_crops）
                    q_small = nn.functional.normalize(q_small, dim=-1)

                # compute key features
                with torch.no_grad():  # no gradient to keys
                    self._momentum_update_key_encoder(base_sess)  # update the key encoder
                    _, k = self.encode_k(im_k)  # keys: bs x dim
                    k = nn.functional.normalize(k, dim=1)

                # compute logits
                # Einstein sum is more intuitive
                # positive logits: Nx1
                q_global = q
                l_pos_global = (q_global * k.unsqueeze(1)).sum(2).view(-1,1)  # 计算q_global和k的点积,再求和压缩为一个数,view成[batch_size, 1]的SHAPE,得到全局图像的positive logits l_pos_global
                l_pos_small = (q_small * k.unsqueeze(1)).sum(2).view(-1, 1)

                # negative logits: NxK
                l_neg_global = torch.einsum('nc,ck->nk', [q_global.view(-1, feat_dim),
                                                          self.queue.clone().detach()])  # self.queue是特征队列,SHAPE是[dim, queue_size],包含历史batch的编码
                l_neg_small = torch.einsum('nc,ck->nk', [q_small.view(-1, feat_dim),
                                                         self.queue.clone().detach()])  # 用einsum计算q_global和self.queue的点积,得到[batch_size, queue_size]的negative logits l_neg_global

                # logits: Nx(1+K)
                logits_global = torch.cat([l_pos_global, l_neg_global],
                                          dim=1)  # logits_global是全局图像的对比学习logits,SHAPE是[batch_size, 1 + queue_size
                logits_small = torch.cat([l_pos_small, l_neg_small], dim=1)

                # apply temperature
                logits_global /= self.T
                logits_small /= self.T

                # one-hot target from augmented image
                positive_target = torch.ones((b, 1)).cuda()  # positive_target是一个全1的tensor,shape是[batch_size, 1],表示query图像自己是positive sample。
                # find same label images from label queue
                # for the query with -1, all
                # targets通过比较labels和队列queue中的标签,找到与query标签相同的样本,标记为1
                targets = ((labels[:, None] == self.label_queue[None, :]) & (labels[:, None] != -1)).float().cuda()
                targets_global = torch.cat([positive_target, targets],
                                           dim=1)  # targets_global将positive_target和targets在dim=1上拼接,得到全局图像的targets。
                targets_small = targets_global.repeat_interleave(repeats=self.args.num_crops[1],
                                                                 dim=0)  # targets_small重复targets_global得到小样本的targets（因为小图像的类别和之前的也是一样的）
                labels_small = labels.repeat_interleave(repeats=self.args.num_crops[1], dim=0)  # labels_small重复labels得到小样本对应的labels。

                # dequeue and enqueue
                if base_sess or (not base_sess and last_epochs_new):
                    self._dequeue_and_enqueue(k, labels)

                return logits_classify, logits_global, logits_small, targets_global, targets_small, txt_ft

        elif self.mode == 'encoder':
            x, _ = self.encode_q(im_cla) # 也是输出池化后的特征
            return x
        else:
            raise ValueError('Unknown mode')



    #还不太清楚具体的目的，因为找不到使用这部分代码的地方
    def update_fc(self,dataloader,class_list,transform,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch] # 加载数据到GPU
            b = data.size()[0] # 获取batch size
            data = transform(data) # 数据增强
            m = data.size()[0] // b # 计算增强次数m
            labels = torch.stack([label*m+ii for ii in range(m)], 1).view(-1) # 生成虚拟类标签
            data, _ =self.encode_q(data) # 通过encoder提取特征
            data.detach() # 分离梯度

        if self.args.not_data_init: # 判断是否随机初始化
            new_fc = nn.Parameter(
                torch.rand(len(class_list)*m, self.num_features, device="cuda"),
                requires_grad=True) # 随机初始化新的fc权重，
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5)) # kaiming初始化
        else:
            new_fc = self.update_fc_avg(data, labels, class_list, m) # 计算类平均特征并更新fc的权重

    # 利用平均特征（平均向量）更新fc的方法
    def update_fc_avg(self,data,labels,class_list, m):
        new_fc=[]
        for class_index in class_list: # 遍历每个新类class_index
            for i in range(m): # 遍历每个变换m
                index = class_index*m + i # 以虚拟类标签为依据计算类索引
                data_index=(labels==index).nonzero().squeeze(-1) # 找到对应index的特征
                embedding=data[data_index] # 取出对应特征
                proto=embedding.mean(0) # 计算该类的平均向量
                new_fc.append(proto) # 将proto作为新的权重保存到new_fc和self.fc中
                self.fc.weight.data[index]=proto # 更新self.fc以新的权重
        new_fc=torch.stack(new_fc,dim=0) # 转换new_fc格式
        return new_fc

    '''
    用于后续计算joint loss
    如果是'dot'模式,直接计算线性logits
    如果是'cos'模式：
        对特征和权重分别做L2归一化
        计算余弦相似度
        调整temperature
        返回logits
        
    get_logits: 输入特征x和全连接层权重fc。
    forward_metric: 输入图像数据x。
    '''

    # 在finetuning阶段，logits是跟有全部seen 类上的全连接层进行计算，所以需要输入包含新类的fc，所以与forward_metric()区别
    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))