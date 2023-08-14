import numpy as np
import torch
import torchvision.transforms as transforms
from dataloader.sampler import CategoriesSampler
from augmentations.constrained_cropping import CustomMultiCropDataset, CustomMultiCropping

# 已读
'''
定义数据增强方式，读取Dataset类
'''
def set_up_datasets(args):
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    args.Dataset=Dataset
    return args
'''
所以图像经过crop_transform生成所需要的大裁剪和小裁剪
然后再将这些裁剪经过随机/默认辅助增强
小裁剪用于query encoder，便于捕获局部信息  
大裁剪用于key encoder，增加多样性
'''
def get_transform(args):
    if args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
    if args.dataset == 'cub200':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]) # 定义cub200的标准化参数
    if args.dataset == 'mini_imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    '''
    crop_transform
    size_large：大裁剪的预期输出尺寸。
    scale_large：大裁剪的原始尺寸范围。
    size_small：小裁剪的预期输出尺寸。
    scale_small：小裁剪的原始尺寸范围。
    N_large：大裁剪的数量。
    N_small：小裁剪的数量。
    ratio：原始长宽比的范围。
    interpolation：默认值为PIL.Image.BILINEAR。
    condition_small_crops_on_key：是否将小裁剪与关键裁剪关联起来。
    '''
    assert (len(args.size_crops) == 2) # 确认裁剪尺寸参数列表为2，如果是则继续
    crop_transform = CustomMultiCropping(size_large=args.size_crops[0],
                                         scale_large=(args.min_scale_crops[0], args.max_scale_crops[0]),
                                         size_small=args.size_crops[1],
                                         scale_small=(args.min_scale_crops[1], args.max_scale_crops[1]),
                                         N_large=args.num_crops[0], N_small=args.num_crops[1],
                                         condition_small_crops_on_key=args.constrained_cropping)

    # 如果没有选择自动增强，则默认一种辅助增强，对所有图片执行
    if len(args.auto_augment) == 0:
        print('No auto augment - Apply regular moco v2 as secondary transform') # 如果没有需要auto裁剪的区域号，则使用常规transform
        secondary_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),    
            transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
#             transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            normalize])

    # 否则往列表中放辅助增强的类型，随机一一增强图片列表
    else:
        from utils.auto_augment.auto_augment import AutoAugment
        from utils.auto_augment.random_choice import RandomChoice
        print('Auto augment - Apply custom auto-augment strategy')
        counter = 0 # counter记录当前是第几个裁剪区域
        secondary_transform = []

        for i in range(len(args.size_crops)):
            for j in range(args.num_crops[i]): # 遍历大和小图片分别需要裁剪出的样本数量
                if not counter in set(args.auto_augment): # 如果counter在指定使用AutoAugment的裁剪序号列表中,则使用trans1。
                    print('Crop {} - Apply regular secondary transform'.format(counter))
                    # 默认裁剪
                    secondary_transform.extend([transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                        ], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
#                         transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                        transforms.ToTensor(),
                        normalize])])

                else: # 否则使用默认的trans2
                    print('Crop {} - Apply auto-augment/regular secondary transform'.format(counter))
                    #
                    trans1 = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        AutoAugment(), # 随机裁剪
                        transforms.ToTensor(),
                        normalize])

                    trans2 = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                        ], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
#                         transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                        transforms.ToTensor(),
                        normalize])

                    secondary_transform.extend([RandomChoice([trans1, trans2])]) # 直接使用append添加会创建嵌套列表:[[trans1], [trans2], [trans3]],extend可以避免嵌套。

                counter += 1
    return crop_transform, secondary_transform

def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args)
    return trainset, trainloader, testloader

def get_base_dataloader(args):
    crop_transform, secondary_transform = get_transform(args)
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class) # 生成0到args.base_class的数组作为类索引，用于后续让对应的数据集loader函数从数据集中正确筛选图像和标签列表dataandtargets
                                             # 然后正确构建Dataset类对象给DataLoader使用
    if args.dataset == 'cifar100':

        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True, index=class_index, base_sess=True,
                                         crop_transform=crop_transform, secondary_transform=secondary_transform)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,index=class_index, base_sess=True,
                                       crop_transform=crop_transform, secondary_transform=secondary_transform)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index=class_index, base_sess=True,
                                             crop_transform=crop_transform, secondary_transform=secondary_transform)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader


def get_new_dataloader(args,session):
    crop_transform, secondary_transform = get_transform(args) # 获取数据增强的crop转换和secondary转换
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt' # 拼接每次session的索引文件路径，指向session文件路径
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines() # 读取txt路径下的待训练的索引类(文件路径)
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False, index=class_index, base_sess=False,                                              crop_transform=crop_transform, secondary_transform=secondary_transform)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True, index_path=txt_path,
                                       crop_transform=crop_transform, secondary_transform=secondary_transform) # 用索引文件路径构建训练集
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index_path=txt_path,
                                             crop_transform=crop_transform, secondary_transform=secondary_transform)
    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__() # 批量大小为训练集长度
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True) # 构建训练数据加载器/有些参数的作用还需验证
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session) # 获取当前session的所有类别

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new) # 构建测试集
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                      index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True) # 构建测试数据加载器

    return trainset, trainloader, testloader

def get_session_classes(args, session):
    class_list=np.arange(args.base_class + session * args.way)
    # 使用numpy的arange函数生成一个数组
    # 数组从0开始,到 args.base_class + session * args.way结束
    # 其中args.base_class是基类的数量
    # args.way是每次新增的class数量
    # session是当前session编号
    return class_list