import os
import numpy as np
from PIL import Image
import random
import logging
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from matplotlib import font_manager
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
import math
from seaborn.distributions import distplot
from tqdm import tqdm
from scipy import ndimage
from utils.get_weak_anns import find_bbox

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.init as initer


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9, index_split=-1, scale_lr=10., warmup=False,
                       warmup_step=500):
    """poly learning rate policy"""
    if warmup and curr_iter < warmup_step:
        lr = base_lr * (0.1 + 0.9 * (curr_iter / warmup_step))
    else:
        lr = base_lr * (1 - float(curr_iter) / max_iter) ** power

    # if curr_iter % 50 == 0:
    #     print('Base LR: {:.4f}, Curr LR: {:.4f}, Warmup: {}.'.format(base_lr, lr, (warmup and curr_iter < warmup_step)))

    for index, param_group in enumerate(optimizer.param_groups):
        if index <= index_split:
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr * scale_lr  # 10x LR


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)


def init_weights(model, conv='xavier', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m,
                        (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):  # , BatchNorm1d, BatchNorm2d, BatchNorm3d)):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


# ------------------------------------------------------
def get_model_para_number(model):
    total_number = 0
    learnable_number = 0
    for para in model.parameters():
        total_number += torch.numel(para)
        if para.requires_grad == True:
            learnable_number += torch.numel(para)
    return total_number, learnable_number


def setup_seed(seed=2021, deterministic=False):
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f'seed is {seed}')


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def get_save_path(args):
    # backbone_str = 'vgg' if args.vgg else 'resnet' + str(args.layers)
    backbone_str = 'vgg'
    args.backbone_path = 'exp/{}/{}/split{}/{}'.format(args.data_set, args.arch, args.split, backbone_str)
    checkpoint_path = args.time
    args.time_path = os.path.join(args.backbone_path, checkpoint_path)
    args.snapshot_path = os.path.join(args.time_path, 'snapshot')
    args.result_path = os.path.join(args.time_path, 'result')
    args.show_path = os.path.join(args.time_path, 'show', 'cluster')
    print(args.time_path)
    print(args.snapshot_path)
    print(args.result_path)
    print(args.show_path)
    # backbone_str = 'vgg' if args.vgg else 'resnet' + str(args.layers)
    # args.snapshot_path = 'exp/{}/{}/split{}/{}/snapshot'.format(args.data_set, args.arch, args.split, backbone_str)
    # args.result_path = 'exp/{}/{}/split{}/{}/result'.format(args.data_set, args.arch, args.split, backbone_str)


def get_train_val_set(args):
    if args.data_set == 'pascal':
        class_list = list(range(1, 21))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        if args.split == 3:
            sub_list = list(range(1, 16))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            sub_val_list = list(range(16, 21))  # [16,17,18,19,20]
        elif args.split == 2:
            sub_list = list(range(1, 11)) + list(range(16, 21))  # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
            sub_val_list = list(range(11, 16))  # [11,12,13,14,15]
        elif args.split == 1:
            sub_list = list(range(1, 6)) + list(range(11, 21))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
            sub_val_list = list(range(6, 11))  # [6,7,8,9,10]
        elif args.split == 0:
            sub_list = list(range(6, 21))  # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            sub_val_list = list(range(1, 6))  # [1,2,3,4,5]

    elif args.data_set == 'coco':
        if args.use_split_coco:
            print('INFO: using SPLIT COCO (FWB)')
            class_list = list(range(1, 81))
            if args.split == 3:
                sub_val_list = list(range(4, 81, 4))
                sub_list = list(set(class_list) - set(sub_val_list))
            elif args.split == 2:
                sub_val_list = list(range(3, 80, 4))
                sub_list = list(set(class_list) - set(sub_val_list))
            elif args.split == 1:
                sub_val_list = list(range(2, 79, 4))
                sub_list = list(set(class_list) - set(sub_val_list))
            elif args.split == 0:
                sub_val_list = list(range(1, 78, 4))
                sub_list = list(set(class_list) - set(sub_val_list))
        else:
            print('INFO: using COCO (PANet)')
            class_list = list(range(1, 81))
            if args.split == 3:
                sub_list = list(range(1, 61))
                sub_val_list = list(range(61, 81))
            elif args.split == 2:
                sub_list = list(range(1, 41)) + list(range(61, 81))
                sub_val_list = list(range(41, 61))
            elif args.split == 1:
                sub_list = list(range(1, 21)) + list(range(41, 81))
                sub_val_list = list(range(21, 41))
            elif args.split == 0:
                sub_list = list(range(21, 81))
                sub_val_list = list(range(1, 21))

    return sub_list, sub_val_list


def is_same_model(model1, model2):
    flag = 0
    count = 0
    for k, v in model1.state_dict().items():
        model1_val = v
        model2_val = model2.state_dict()[k]
        if (model1_val == model2_val).all():
            pass
        else:
            flag += 1
            print('value of key <{}> mismatch'.format(k))
        count += 1

    return True if flag == 0 else False


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def sum_list(list):
    sum = 0
    for item in list:
        sum += item
    return sum


def memory_bank_upload(pro_list, lenth, cls, memory_bank, ptr):
    """
    :param pro_list: [bs, 256, shot, h, w]
    :param lenth: bank长度
    :param cls: 标签 = cat_ids
    :param memory_bank: bank实例[categroies, lenth, channels, H, W]
    :param ptr: 指针--1维度tensor  一开始全0的15维计数器
    :return:memory_bank, ptr, mean_pro, loss_memo

    cls[0][i] 标签名
    supp_pro[2] shot数
    memory_bank[标签， 数量， 256， h， w]
    """
    # supp_pro = torch.tensor(torch.stack(pro_list, 2))  # 在第三个维度（shot维度）拼接但是不扩充维度
    supp_pro = torch.stack(pro_list, 2)  # 在第三个维度（shot维度）拼接但是不扩充维度
    # print('supp_pro shape={}'.format(supp_pro.shape))
    for i in range(supp_pro.shape[0]):  # 在第一个维度(bs)循环
        if (ptr[cls[0][i]] + supp_pro.shape[2]) < lenth:  # 如果计数器和supp_pro数量维度之和不超过bank长度
            memory_bank[cls[0][i], int(ptr[cls[0][i]]):int(ptr[cls[0][i]]) + supp_pro.shape[2], :, :, :] = \
                supp_pro[i, :, :, :, :].permute(1, 0, 2, 3)  # 调整维度[bs, 256, shot, h, w]->[bs, shot, 256, h, w]
            # memory_bank[标签， 数量->数量+supp_pro的shot数， 256， h， w] = supp_pro[bs, shot, 256, h, w]
            ptr[cls[0][i]] += supp_pro.shape[2]  # 计数器计数
        else:
            # 如果计数器和supp_pro数量维度之和超过bank长度
            memory_bank[cls[0][i], int(ptr[cls[0][i]]):lenth, :, :, :] = \
                supp_pro[i, :, :, :, :].permute(1, 0, 2, 3)[:(lenth - int(ptr[cls[0][i]])), :, :, :]
            # memory_bank[标签， 数量->上限， 256， h， w] = supp_pro[bs, 从头->上限-数量, 256, h, w]
            memory_bank[cls[0][i], :(supp_pro.shape[2] - (lenth - int(ptr[cls[0][i]]))), :, :, :] = \
                supp_pro[i, :, :, :, :].permute(1, 0, 2, 3)[(lenth - int(ptr[cls[0][i]])):, :, :, :]
            # memory_bank[标签， 开头->差额， 256， h， w] = supp_pro[bs, 上限-数量->结束, 256, h, w]
            ptr[cls[0][i]] = supp_pro.shape[2] - (lenth - int(ptr[cls[0][i]]))  # 计数

    return memory_bank, ptr


def add_conf_param(weight, conf_param, x):
    b = weight[:, -1, :, :] * conf_param ** x
    weight.clone()[:, -1, :, :] = b
    return weight.softmax(dim=1)


def conf_filter_per(tensor, filter_param0, filter_param1, black):
    """
    :param filter_param1:
    :param filter_param0:
    :param black:
    :param tensor:[2, h, w]
    :return: [h, w]
    """
    confidence = tensor[1] - tensor[0]
    pre_filter0 = torch.zeros_like(confidence)
    pre_filter1 = torch.zeros_like(confidence)
    pre_filter0[confidence < 0] = 1
    pre_filter0[black == 255] = 0
    pre_filter1[confidence > 0] = 1
    pre_filter1[black == 255] = 0

    ori_down0 = tensor.clone()[0]
    filt_down0 = torch.where(pre_filter0 == 1, ori_down0, torch.zeros_like(ori_down0))
    down0 = filt_down0.view(1, -1)
    down_tensor0, indices0 = torch.sort(down0)
    no_zero_down0 = torch.nonzero(down_tensor0)
    idx0 = int(no_zero_down0.shape[0] * filter_param0)

    ori_down1 = tensor.clone()[1]
    filt_down1 = torch.where(pre_filter1 == 1, ori_down1, torch.zeros_like(ori_down1))
    down1 = filt_down1.view(1, -1)
    down_tensor1, indices1 = torch.sort(down1)
    no_zero_down1 = torch.nonzero(down_tensor1)
    idx1 = int(no_zero_down1.shape[0] * filter_param1)

    # if idx0 > round(idx1 * 9):
    #     idx0 = round(idx1 * 9)

    # print(idx)
    # print(down_tensor, indices)
    stand0 = down_tensor0[0, -idx0] + 1e-10
    # print(stand)
    conf_matrix0 = torch.zeros_like(tensor[0])
    conf_matrix0[tensor[0] >= stand0] = 1

    # print(idx)
    # print(down_tensor, indices)
    stand1 = down_tensor1[0, -idx1] + 1e-10
    # print(stand)
    conf_matrix1 = torch.zeros_like(tensor[1])
    conf_matrix1[tensor[1] >= stand1] = 1
    # print(conf_matrix)
    conf_matrix = conf_matrix0 + conf_matrix1
    return conf_matrix, conf_matrix0, conf_matrix1


def get_conf_matrix(output, filter_param0, filter_param1, black_record):
    """
    :param filter_param1:
    :param filter_param0:
    :param black_record:
    :param output:[bs, 2, h, w]
    :return:
    """
    bs = output.shape[0]
    result_list = []
    result_list0 = []
    result_list1 = []
    for i in range(bs):
        # if ft_round == 0:
        result, result0, result1 = conf_filter_per(output[i], filter_param0, filter_param1, black_record[i])
        result_list.append(result)
        result_list0.append(result0)
        result_list1.append(result1)
        # result[result >= 1] = 1
        conf_matrix = torch.stack(result_list, dim=0)
        conf_matrix0 = torch.stack(result_list0, dim=0)
        conf_matrix1 = torch.stack(result_list1, dim=0)
        # conf_param = torch.mean(conf_matrix, dim=(-2, -1)).unsqueeze(dim=-1)
        return conf_matrix, conf_matrix0, conf_matrix1
        # elif ft_round > 0:
        #     result, result0, result1 = conf_filter_fix(output[i], filter_stand0, filter_stand1)
        #     result_list.append(result)
        #     result_list0.append(result0)
        #     result_list1.append(result1)
        #     # result[result >= 1] = 1
        #     conf_matrix = torch.stack(result_list, dim=0)
        #     conf_matrix0 = torch.stack(result_list0, dim=0)
        #     conf_matrix1 = torch.stack(result_list1, dim=0)
        #     # conf_param = torch.mean(conf_matrix, dim=(-2, -1)).unsqueeze(dim=-1)
        #     return conf_matrix, conf_matrix0, conf_matrix1


def conf_filter_fix(tensor, filter_stand0, filter_stand1):
    """
    :param filter_stand1:
    :param filter_stand0:
    :param tensor:[2, h, w]
    :return: [h, w]
    """

    stand0 = filter_stand0
    conf_matrix0 = torch.zeros_like(tensor[0])
    conf_matrix0[tensor[0] >= stand0] = 1

    stand1 = filter_stand1
    conf_matrix1 = torch.zeros_like(tensor[1])
    conf_matrix1[tensor[1] >= stand1] = 1
    conf_matrix = conf_matrix0 + conf_matrix1
    return conf_matrix, conf_matrix0, conf_matrix1


def ft_learning_rate(optimizer, base_lr, ft_num, ft_now, power=1):
    """poly learning rate policy"""
    lr = base_lr * (1 - ft_now / (ft_num + 1)) ** power + 1e-10
    # if curr_iter % 50 == 0:
    #     print('Base LR: {:.4f}, Curr LR: {:.4f}, Warmup: {}.'.format(base_lr, lr, (warmup and curr_iter < warmup_step)))

    for index, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr


def show_img(base, label, mode, path, color):
    """
    :param color:
    :param base: 基础图片[bs, 3, h, w]
    :param label: 标签[1, h, w]
    :param mode: 重叠模式或者单独图片 'cat' or 'sin'
    :param path: 保存路径
    :return:
    """
    assert len(base.shape) == 4, len(label.shape) == 3
    assert base.shape[-2:] == label.shape[-2:]
    if mode == 'cat':
        base_show = base.clone()
        base_show = ((base_show - torch.min(base_show)) / (
                torch.max(base_show) - torch.min(base_show)) * 255).type(torch.int)
        input_np = base_show[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        input_img = Image.fromarray(np.uint8(input_np))
        target_tensor = label.clone()
        targetr = torch.zeros_like(target_tensor)
        targetg = torch.zeros_like(target_tensor)
        targetb = torch.zeros_like(target_tensor)
        if color == 'r':
            targetr[target_tensor >= 1] = 255
            targetr[target_tensor == 255] = 0
        if color == 'g':
            targetg[target_tensor >= 1] = 255
            targetg[target_tensor == 255] = 0
        if color == 'b':
            targetb[target_tensor >= 1] = 255
            targetb[target_tensor == 255] = 0
        target_tensor = torch.cat([targetr, targetg, targetb], dim=0)
        target_np = target_tensor.permute(1, 2, 0).detach().cpu().numpy()
        target_img = Image.fromarray(np.uint8(target_np))
        mask_img_show = Image.blend(input_img.convert('RGBA'), target_img.convert('RGBA'), 0.3)
        plt.imshow(mask_img_show)
        plt.savefig(path)
    if mode == 'sin':
        base_show = ((base - torch.min(base)) / (
                torch.max(base) - torch.min(base)) * 255).type(torch.int)
        input_np = base_show[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        input_img = Image.fromarray(np.uint8(input_np))
        plt.imshow(input_img)
        plt.savefig(path)
    if mode == 'mask':
        target_tensor = label.clone()
        targetr = torch.zeros_like(target_tensor)
        targetg = torch.zeros_like(target_tensor)
        targetb = torch.zeros_like(target_tensor)
        targetr[target_tensor >= 1] = 255
        targetr[target_tensor == 255] = 0
        targetg[target_tensor >= 1] = 255
        targetg[target_tensor == 255] = 0
        targetb[target_tensor >= 1] = 255
        targetb[target_tensor == 255] = 0
        target_tensor = torch.cat([targetr, targetg, targetb], dim=0)
        target_np = target_tensor.permute(1, 2, 0).detach().cpu().numpy()
        target_img = Image.fromarray(np.uint8(target_np))
        plt.imshow(target_img)
        plt.savefig(path)
