# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F

from losses import SupContrastive


def base_train(model, trainloader, criterion, optimizer, scheduler, epoch, transform, args):
    '''
    数据准备:
        batch中包含原图original,两个增强视图data[1], data[2]。
        如果使用多crop策略,还包含小尺寸裁剪数据data_small。
        将数据加载到GPU。
    数据增强:
        对原图original做增强,得到data_classify。
        对data[1]做增强,得到query图像data_query。
        对data[2]做增强,得到key图像data_key。
        对data_small也做增强。
    forward计算:
        输入data_classify到模型获取分类预测logits。
        输入data_query获取query特征,计算全局对比loss。
        输入小尺寸crop data_small获取小尺寸query特征,计算小尺寸对比loss。
    loss计算:
        joint_loss是分类loss,交叉熵损失。
        loss_moco_global是全局对比学习loss。
        loss_moco_small是小尺寸crop的对比学习loss。
        汇总不同loss,平衡权重,得到总loss。
    优化更新:
        清零梯度,反向传播,更新参数。
    记录日志:
        计算分类准确率acc。
        打印loss和acc到日志。
        更新平均loss和acc。
        返回loss和acc供后续使用。
    '''
    # 初始化Averager对象
    tl = Averager()  # 用于累加总损失并计算平均值。
    tl_joint = Averager()  # 用于累加joint loss并计算平均值。
    tl_moco = Averager()  # 用于累加moco总损失并计算平均值。
    tl_moco_global = Averager()  # 用于累加moco全局损失并计算平均值。
    tl_moco_small = Averager()  # 用于累加moco小尺寸损失并计算平均值。
    ta = Averager()  # 用于累加准确率并计算平均值。
    model = model.train()
    tqdm_gen = tqdm(trainloader)  # 使用tqdm包tqdm_gen获取dataloader的进度条迭代器,用于在训练时显示进度条。

    # criterion_txt = nn.CrossEntropyLoss()
    for i, batch in enumerate(tqdm_gen, 1):  # 循环遍历dataloader,同时显示进度条。
        data, single_labels = [_ for _ in batch]  # 从batch中取出data和标签。
        b, c, h, w = data[1].shape  # 获取data[1]的shape信息。
        original = data[0].cuda(non_blocking=True)  # 将原始图片original加载到GPU。作用是用于分类
        data[1] = data[1].cuda(non_blocking=True)  # 将增强图片1加载到GPU。
        data[2] = data[2].cuda(non_blocking=True)  # 将增强图片2加载到GPU。
        single_labels = single_labels.cuda(non_blocking=True)  # 将标签加载到GPU。
        if len(args.num_crops) > 1:  # 如果使用了多crop策略:
            data_small = data[args.num_crops[0] + 1].unsqueeze(1)  # 构建小尺寸裁剪图片data_small
            for j in range(1, args.num_crops[1]):
                data_small = torch.cat((data_small, data[j + args.num_crops[0] + 1].unsqueeze(1)), dim=1)
            data_small = data_small.view(-1, c, args.size_crops[1], args.size_crops[1]).cuda(
                non_blocking=True)  # 调整shape为[b * num_small_crops, c, h, w],然后加载到GPU。
        else:
            data_small = None

        data_classify = transform(original)
        data_query = transform(data[1])  # 生成新的虚拟类
        data_key = transform(data[2])
        data_small = transform(data_small)
        m = data_query.size()[0] // b
        joint_labels = torch.stack([single_labels * m + ii for ii in range(m)], 1).view(-1)

        if args.no_semantic:
            joint_preds, output_global, output_small, target_global, target_small = model(
                im_cla=data_classify,
                im_q=data_query, im_k=data_key,
                labels=joint_labels,
                im_q_small=data_small)
            # print(joint_labels, encoded_text_features)
            # print(encoded_text_features.shape, joint_labels.shape)

            # joint_preds, output_global, output_small, target_global, target_small = model(im_cla=data_classify, im_q=data_query, im_k=data_key, labels=joint_labels, im_q_small=data_small)
            loss_moco_global = criterion(output_global, target_global)
            loss_moco_small = criterion(output_small, target_small)
            loss_moco = args.alpha * loss_moco_global + args.beta * loss_moco_small

            joint_preds = joint_preds[:, :args.base_class * m]  # model的分类头self.fc默认是输出num_classes个预测的,不仅仅是base_class类。
            joint_loss = F.cross_entropy(joint_preds, joint_labels)

            agg_preds = 0
            for i in range(m):
                agg_preds = agg_preds + joint_preds[i::m, i::m] / m

            # semantic_loss和joint_loss没经过同一个fc层
            loss = joint_loss + loss_moco

        else:
            # 加入文本编码器咯
            if args.fantasy == 'rotation2':
                # 定义通过类别序列生成语义信息
                classes_file = '/kaggle/input/data-file-for-svtc/data/CUB_200_2011/classes.txt'
                # 从类别文本文件中加载类别id到名称的映射
                with open(classes_file) as f:
                    classes_txt = f.read()

                classes = {}
                for line in classes_txt.split('\n'):
                    if line.strip() == '':
                        continue
                    label, name = line.split(' ')
                    classes[int(label) - 1] = name.split('.', 1)[-1].replace('_', ' ')

            # "A photo about dolphins." "A photo sourced from dolphins." "A 190 degree rotated photo about dolphins."
            semantic_text = []
            for i in range(len(joint_labels)):
                if i % 2 == 0:
                    label_index = joint_labels[i].item() // m
                    classname = classes[label_index]
                    text = f'A image about {classname}'
                else:
                    label_index = (joint_labels[i].item() - 1) // m
                    classname = classes[label_index]
                    text = f'A 190 degree rotated image about {classname}'
                semantic_text.append(text)
            # print(semantic_text)

            # 如果不使用文本信息则修改下面这两行
            model.mode = 'semantic'
            joint_preds, output_global, output_small, target_global, target_small, encoded_text_features = model(
                im_cla=data_classify,
                im_q=data_query, im_k=data_key,
                labels=joint_labels,
                im_q_small=data_small, txt=semantic_text)
            # print(joint_labels, encoded_text_features)
            # print(encoded_text_features.shape, joint_labels.shape)

            semantic_loss = F.cross_entropy(encoded_text_features, joint_labels)

            # joint_preds, output_global, output_small, target_global, target_small = model(im_cla=data_classify, im_q=data_query, im_k=data_key, labels=joint_labels, im_q_small=data_small)
            loss_moco_global = criterion(output_global, target_global)
            loss_moco_small = criterion(output_small, target_small)
            loss_moco = args.alpha * loss_moco_global + args.beta * loss_moco_small

            joint_preds = joint_preds[:, :args.base_class * m]  # model的分类头self.fc默认是输出num_classes个预测的,不仅仅是base_class类。
            joint_loss = F.cross_entropy(joint_preds, joint_labels)

            agg_preds = 0
            for i in range(m):
                agg_preds = agg_preds + joint_preds[i::m, i::m] / m

            # semantic_loss和joint_loss没经过同一个fc层
            loss = args.lamda * semantic_loss + (1 - args.lamda) * joint_loss + loss_moco

        total_loss = loss

        acc = count_acc(agg_preds, single_labels)

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        tl_joint.add(joint_loss.item())
        tl_moco_global.add(loss_moco_global.item())
        tl_moco_small.add(loss_moco_small.item())
        tl_moco.add(loss_moco.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    tl_joint = tl_joint.item()
    tl_moco = tl_moco.item()
    tl_moco_global = tl_moco_global.item()
    tl_moco_small = tl_moco_small.item()
    return tl, tl_joint, tl_moco, tl_moco_global, tl_moco_small, ta


# '''
# 这段代码的主要作用是用训练集数据替换模型中fc层的权重,具体做法是:
#     用训练集DataLoader批量提取所有训练数据的特征向量embedding。
#     根据样本标签label_list将特征向量embedding划分到不同类。
#     对每个类计算特征向量的均值prototype。
#     将计算出的prototype作为新的fc权重,替换原来的fc.weight。
# 这样做的效果是:
#     原来fc层的权重是随机初始化的。
#     替换为根据训练集特征计算出的类均值prototype。
#     新的fc权重更接近真实样本的特征分布。
# 代替了随机初始化的权重。
# 所以这段代码实现了用训练数据来更新fc层权重的效果,使新的fc权重更好地反映训练数据的特征分布,起到一个精炼原始随机权重的作用。
# 是典型的用训练数据来更新模型权重的做法,在迁移学习等任务中很常见。
# '''
def replace_base_fc(trainset, test_transform, data_transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=2, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = test_transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = data_transform(data)
            m = data.size()[0] // b
            labels = torch.stack([label * m + ii for ii in range(m)], 1).view(-1)
            model.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(labels.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class * m):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.fc.weight.data[:args.base_class * m] = proto_list

    return model


# incremental learning finetuning 如果效果不好可以再解冻一层试试
def update_fc_ft(trainloader, data_transform, model, m, session, args):
    # incremental finetuning
    old_class = args.base_class + args.way * (session - 1)
    new_class = args.base_class + args.way * session
    new_fc = nn.Parameter(
        torch.rand(args.way * m, model.num_features, device="cuda"),
        requires_grad=True)
    new_fc.data.copy_(model.fc.weight[old_class * m: new_class * m,
                      :].data)  # 定义新的线性分类层new_fc,其权重初始化为之前旧类的权重。这样可以加速新的类的训练,利用旧类的先验知识。更新的是predictor的参数用于分类
    # 这个new_fc相当于fintuning阶段使用的predictor

    if args.dataset == 'mini_imagenet':
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new},
                                     {'params': model.encoder_q.fc.parameters(), 'lr': 0.05 * args.lr_new},
                                     {'params': model.encoder_q.layer4.parameters(), 'lr': 0.001 * args.lr_new}, ],
                                    momentum=0.9, dampening=0.9, weight_decay=0)

    if args.dataset == 'cub200':  # 样本小所以只更新predictor，这很好理解
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new}],
                                      # {'params': model.encoder_q.fc.parameters(), 'lr': 0.01 * args.lr_new},
                                     weight_decay = args.decay_new)  # 可以尝试像imagenet的策略一样冻结
                                     # momentum=0.9, dampening=0.9, weight_decay=0)  # 可以尝试像imagenet的策略一样冻结

    elif args.dataset == 'cifar100':
        optimizer = torch.optim.Adam([{'params': new_fc, 'lr': args.lr_new},
                                      {'params': model.encoder_q.fc.parameters(), 'lr': 0.01 * args.lr_new},
                                      {'params': model.encoder_q.layer3.parameters(), 'lr': 0.02 * args.lr_new}],
                                     weight_decay=0)

    criterion = SupContrastive().cuda()

    with torch.enable_grad():
        for epoch in range(args.epochs_new):
            for batch in trainloader:
                data, single_labels = [_ for _ in batch]
                b, c, h, w = data[1].shape
                origin = data[0].cuda(non_blocking=True)
                data[1] = data[1].cuda(non_blocking=True)
                data[2] = data[2].cuda(non_blocking=True)
                single_labels = single_labels.cuda(non_blocking=True)
                if len(args.num_crops) > 1:
                    data_small = data[args.num_crops[0] + 1].unsqueeze(1)
                    for j in range(1, args.num_crops[1]):
                        data_small = torch.cat((data_small, data[j + args.num_crops[0] + 1].unsqueeze(1)), dim=1)
                    data_small = data_small.view(-1, c, args.size_crops[1], args.size_crops[1]).cuda(non_blocking=True)
                else:
                    data_small = None
            data_classify = data_transform(origin)
            data_query = data_transform(data[1])
            data_key = data_transform(data[2])
            data_small = data_transform(data_small)
            joint_labels = torch.stack([single_labels * m + ii for ii in range(m)], 1).view(-1)

            old_fc = model.fc.weight[:old_class * m, :].clone().detach()
            fc = torch.cat([old_fc, new_fc], dim=0)  # 整合新旧fc在一起
            features, _ = model.encode_q(data_classify)
            features.detach()
            logits = model.get_logits(features, fc)  # 在所有已见类上计算logits用于分类crossentropy
            joint_loss = F.cross_entropy(logits, joint_labels)
            _, output_global, output_small, target_global, target_small = model(im_cla=data_classify, im_q=data_query,
                                                                                im_k=data_key, labels=joint_labels,
                                                                                im_q_small=data_small, base_sess=False,
                                                                                last_epochs_new=(
                                                                                            epoch == args.epochs_new - 1))
            loss_moco_global = criterion(output_global, target_global)
            loss_moco_small = criterion(output_small, target_small)
            loss_moco = args.alpha * loss_moco_global + args.beta * loss_moco_small
            loss = joint_loss + loss_moco
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.fc.weight.data[old_class * m: new_class * m, :].copy_(new_fc.data)  # 更新这一轮得到的新的权重为下一轮旧的全fc权重了


def test(model, testloader, epoch, transform, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = transform(data)
            m = data.size()[0] // b
            joint_preds = model(data)
            joint_preds = joint_preds[:, :test_class * m]

            agg_preds = 0
            # 对各个变换结果聚合预测真实标签
            for j in range(m):
                agg_preds = agg_preds + joint_preds[j::m, j::m] / m

            loss = F.cross_entropy(agg_preds, test_label)
            acc = count_acc(agg_preds, test_label)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va