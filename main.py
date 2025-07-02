import matplotlib.pyplot as plt
import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.Generate_Model_v5 import GenerateModel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
import datetime
from dataloader.video_dataloader import train_data_loader, test_data_loader
from sklearn.metrics import confusion_matrix
import tqdm
from clip import clip
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from models.Text_last import *

import random
from torch.utils.data import WeightedRandomSampler

class LSR2(nn.Module):
    def __init__(self,e,label_mode):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.label_mode = label_mode

    def _one_hot(self, labels, classes, value=1):
        one_hot = torch.zeros(labels.size(0), classes)
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)
        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)
        one_hot.scatter_add_(1, labels, value_added)
        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        mask = (one_hot==0)

        # todo
        if self.label_mode == 'four':
            balance_weight = torch.tensor([0.0510725,2.1551724,2.985074,7.35294117]).to(one_hot.device)
        elif self.label_mode == 'six':
            balance_weight = torch.tensor([0.065267810,0.817977729,1.035884371,0.388144355,0.19551041668]).to(one_hot.device)

        ex_weight = balance_weight.expand(one_hot.size(0),-1)
        resize_weight = ex_weight[mask].view(one_hot.size(0),-1)
        resize_weight /= resize_weight.sum(dim=1, keepdim=True)
        one_hot[mask] += (resize_weight*smooth_factor).view(-1)
        return one_hot.to(target.device)
    
    def forward(self, x, target):
        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)
        return torch.mean(loss)

from torch.distributions import normal
import torch.nn.functional as F
class BlvLoss(nn.Module):
    def __init__(self, cls_num_list, sigma=4, loss_name='BlvLoss'):
        super(BlvLoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        frequency_list = torch.log(cls_list)
        self.frequency_list = torch.log(sum(cls_list)) - frequency_list
        self.reduction = 'mean'
        self.sampler = normal.Normal(0, sigma)
        self._loss_name = loss_name

    def forward(self, pred, target):
        viariation = self.sampler.sample(pred.shape).clamp(-1, 1).to(pred.device)
        pred = pred + (viariation.abs() / self.frequency_list.max() * self.frequency_list)
        loss = F.cross_entropy(pred, target, reduction='none')

        return loss.mean()

parser = argparse.ArgumentParser()
# 基础设置(数据集、显卡、workers)
parser.add_argument('--dataset', type=str,default='AcademicEmotion_five')
parser.add_argument('--gpu', type=int,default = 3)
parser.add_argument('--workers', type=int, default=4)

# 训练设置(实验名称、数据增强、学习率、训练轮数、batch_size、weight_decay、momentum、milestones)
parser.add_argument('--input-type', type=str,default='face')  # face:只有人脸 body:只有身体 all:整张图片
parser.add_argument('--exper-name', type=str,default='test')
parser.add_argument('--data-aug', type=str,default='False')
parser.add_argument('--aug-file', type=str,default='')
parser.add_argument('--cut-body', type=str,default='False')
# 标签平滑
parser.add_argument('--label-smoothing',type=str,default = 'False')
parser.add_argument('--smooth-param',type=float,default = 0.1) 
# 是否进行恢复训练
parser.add_argument('--resume', type=str,default='False')
parser.add_argument('--resume-path', type=str,default='') 
# parser.add_argument('--resume-set', type=str,default='')

parser.add_argument('--set', type=int, default=5)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lr-image-encoder', type=float, default=1e-5)
parser.add_argument('--lr-prompt-learner', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--milestones', nargs='+', type=int,default={10,15})

parser.add_argument('--contexts-number', type=int, default=8)
parser.add_argument('--class-token-position', type=str, default="end")
parser.add_argument('--class-specific-contexts', type=str, default='True')
parser.add_argument('--text-type', type=str,default='class_descriptor')

parser.add_argument('--temporal-layers', type=int, default=1)

# 随机种子设置
parser.add_argument('--seed', type=int,default=42)

# load_and_tune_prompt_learner(是否微调提示学习)
parser.add_argument('--load_and_tune_prompt_learner', type=str, default='True')
args = parser.parse_args()

# 选择设备
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
args.device = device

random.seed(args.seed)  
np.random.seed(args.seed) 
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# 结果存储设置
now = datetime.datetime.now()
time_str = now.strftime("-[%m-%d]-[%H:%M]")
args.name = args.exper_name + str(time_str) 

if args.resume == 'True':
    args.name = args.resume_path
args.output_path = "outputs/" + args.name + "/"

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

print('************************')
for k, v in vars(args).items():
    print(k,'=',v)
print('************************')


if args.dataset == "AcademicEmotion_five":
    number_class = 5
    class_names = [
    'Neutrality',
    'Enjoyment',
    'Confusion',
    'Fatigue',
    'Distraction.'
    ]
    class_names_with_context = class_names_with_context_5
    class_descriptor = class_descriptor_5

elif args.dataset == "AcademicEmotion_four":
    pass

def main():
    if args.dataset == "AcademicEmotion_five":
        print("*********** AcademicEmotion_six Dataset ***********")
        log_txt_path = args.output_path + 'log.txt'
        log_curve_path = args.output_path  + 'log.png'
        log_confusion_matrix_path = args.output_path + 'cnconfusion_matrix.png'
        checkpoint_path = args.output_path + 'model.pth'
        best_checkpoint_path = args.output_path  + 'model_best.pth'
        train_annotation_file_path = "/media/F/FERDataset/AER-DB/new_confusion/label/AcademicEmotion_train_set_5_new.txt"
        test_annotation_file_path = "/media/F/FERDataset/AER-DB/new_confusion/label/AcademicEmotion_test_set_5_new.txt"

    elif args.dataset == "AcademicEmotion_four":
        print("*********** AcademicEmotion_four Dataset ***********")
        # AcademicEmotion1_test_set_1
        log_txt_path = args.output_path  + 'log.txt'
        log_curve_path = args.output_path + 'log.png'
        log_confusion_matrix_path = args.output_path+'confusion_matrix.png'
        checkpoint_path = args.output_path + 'model.pth'
        best_checkpoint_path = args.output_path +  'model_best.pth'
        # 这里可以换位比较固定的数据集标签路径，而不是一份代码一个
        train_annotation_file_path = "/media/F/FERDataset/AER-DB/EmoLabel_DataSplit/four/four_train.txt"
        test_annotation_file_path = "/media/F/FERDataset/AER-DB/EmoLabel_DataSplit/four/four_test.txt"

    best_acc = 0
    recorder = RecorderMeter(args.epochs)
    print('The training name: ' + time_str)

    # create model and load pre_trained parameters,这里填写一个下载好的具体路径
    CLIP_model, _ = clip.load("/media/D/zlm/code/single_four/models/ViT-B-32.pt", device='cpu')
    
    if args.text_type=="class_names":
        input_text = class_names
    elif args.text_type=="class_names_with_context":
        input_text = class_names_with_context
    elif args.text_type=="class_descriptor":
        input_text = class_descriptor

    print("Input Text: ")
    for i in range(len(input_text)):
        print(input_text[i])
        # 写入txt文件
        with open(log_txt_path, 'a') as f:
            f.write(input_text[i] + '\n')
    model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)
    

    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "image_encoder" in name:
            param.requires_grad = True  
        if "temporal_net" in name:
            param.requires_grad = True
        if "prompt_learner" in name:  
            param.requires_grad = True
        if "temporal_net_body" in name:  
            param.requires_grad = True
        if "project_fc" in name:  
            param.requires_grad = True
    # model = model.cuda(1)
    model = model.to(device)

    # print params
    print('************************')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('************************')


    
    
    with open(log_txt_path, 'a') as f:
        for k, v in vars(args).items():
            f.write(str(k) + '=' + str(v) + '\n')

    if args.label_smoothing == 'True':
        if args.dataset == "AcademicEmotion_five":
            criterion = LSR2(e=args.smooth_param, label_mode='six').to(device)
        elif args.dataset == "AcademicEmotion_four":
            criterion = LSR2(e=args.smooth_param, label_mode='four').to(device)
    elif args.label_smoothing == 'False':
        criterion = nn.CrossEntropyLoss().to(device)


    # define optimizer
    optimizer = torch.optim.SGD([{"params": model.temporal_net.parameters(), "lr": args.lr},
                                 {"params": model.temporal_net_body.parameters(), "lr": args.lr},
                                 {"params": model.image_encoder.parameters(), "lr": args.lr_image_encoder},
                                 {"params": model.prompt_learner.parameters(), "lr": args.lr_prompt_learner},
                                 {"params": model.project_fc.parameters(), "lr": args.lr_image_encoder}
                                 ],   ### 合并模块
                                 lr=args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)
    

    # define scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.milestones,
                                                     gamma=0.1)
        
    cudnn.benchmark = True

    # Data loading code
    train_data = train_data_loader(list_file=train_annotation_file_path,
                                   num_segments=16,
                                   duration=1,
                                   image_size=224,
                                   over_sample=False,
                                   args=args)
    test_data = test_data_loader(list_file=test_annotation_file_path,
                                 num_segments=16,
                                 duration=1,
                                 image_size=224,
                                 over_sample= False,
                                 )

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)
    
    if args.resume == 'True':
        checkpoint = torch.load(checkpoint_path,map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        recorder = checkpoint.get('recorder', None)
        epoch = checkpoint['epoch']
    epoch = 0

    for epoch in range(0, args.epochs):
        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate_0 = optimizer.state_dict()['param_groups'][0]['lr']
        current_learning_rate_1 = optimizer.state_dict()['param_groups'][1]['lr']
        current_learning_rate_2 = optimizer.state_dict()['param_groups'][2]['lr']

        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            print(inf)
            f.write('Current learning rate: ' + str(current_learning_rate_0) + ' ' + str(current_learning_rate_1) + ' ' + str(current_learning_rate_2) + '\n')
            print('Current learning rate: ', current_learning_rate_0, current_learning_rate_1, current_learning_rate_2)         
            
        # train for one epoch
        train_acc, train_los,war,uar = train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path)

        # evaluate on validation set
        val_acc, val_los,val_war,val_uar = validate(val_loader, model, criterion, args, log_txt_path)
        
        scheduler.step()

        # remember best acc and save checkpoint
        is_best = val_uar > best_acc
        best_acc = max(val_uar, best_acc)
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best,
                        checkpoint_path,
                        best_checkpoint_path)

        # print and save log
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder.plot_curve(log_curve_path)

        print('The best accuracy: {:.3f}'.format(best_acc.item()))
        print('An epoch time: {:.2f}s'.format(epoch_time))
        with open(log_txt_path, 'a') as f:
            f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
            f.write('An epoch time: ' + str(epoch_time) + 's' + '\n')
    uar, war = computer_uar_war(val_loader, model, best_checkpoint_path, log_confusion_matrix_path, log_txt_path)
    return uar, war


def train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch),
                             log_txt_path=log_txt_path)

    # switch to train mode
    model.train()
    correct = 0
    for i, (images_face,images_body, target) in enumerate(train_loader):

        images_face = images_face.to(device)
        images_body = images_body.to(device)
        target = target.to(device)
        
        # compute output
        output = model(images_face,images_body) 
        predicted = output.argmax(dim=1, keepdim=True)
        correct += predicted.eq(target.view_as(predicted)).sum().item()
        if i == 0:
            all_predicted = predicted
            all_targets = target
        else:
            all_predicted = torch.cat((all_predicted, predicted), 0)
            all_targets = torch.cat((all_targets, target), 0)   
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), images_face.size(0))
        top1.update(acc1[0], images_face.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)
    war = 100. * correct / len(train_loader.dataset)
    # Compute confusion matrix
    _confusion_matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    np.set_printoptions(precision=4)
    normalized_cm = _confusion_matrix.astype('float') / _confusion_matrix.sum(axis=1)[:, np.newaxis]
    normalized_cm = normalized_cm * 100
    list_diag = np.diag(normalized_cm)
    uar = list_diag.mean()   #对角线的位置除以N,即取平均值
    with open(log_txt_path, 'a') as f:
        f.write('Current Accuracy: {top1.avg:.3f}'.format(top1=top1) + '\n')
        f.write('Current UAR: {war:.3f}'.format(war=war) + '\n')
        f.write('Current UAR: {uar:.3f}'.format(uar=uar) + '\n')
    return top1.avg, losses.avg,war,uar


def validate(val_loader, model, criterion, args, log_txt_path):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ',
                             log_txt_path=log_txt_path)

    # switch to evaluate mode
    model.eval()
    correct = 0
    with torch.no_grad():
        for i, (images_face,images_body, target) in enumerate(val_loader):
            
            images_face = images_face.to(device)
            images_body = images_body.to(device)

            target = target.to(device)

            # compute output
            output = model(images_face,images_body) 
            predicted = output.argmax(dim=1, keepdim=True)
            correct += predicted.eq(target.view_as(predicted)).sum().item()
            if i == 0:
                all_predicted = predicted
                all_targets = target
            else:
                all_predicted = torch.cat((all_predicted, predicted), 0)
                all_targets = torch.cat((all_targets, target), 0)              
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), images_face.size(0))
            top1.update(acc1[0], images_face.size(0))

            if i % args.print_freq == 0:
                progress.display(i)
        war = 100. * correct / len(val_loader.dataset)
        # Compute confusion matrix
        _confusion_matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
        np.set_printoptions(precision=4)
        normalized_cm = _confusion_matrix.astype('float') / _confusion_matrix.sum(axis=1)[:, np.newaxis]
        normalized_cm = normalized_cm * 100
        list_diag = np.diag(normalized_cm)
        uar = list_diag.mean()   #对角线的位置除以N,即取平均值
        # TODO: this should also be done with the ProgressMeter
        print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        with open(log_txt_path, 'a') as f:
            f.write('Current Accuracy: {top1.avg:.3f}'.format(top1=top1) + '\n')
            f.write('Current Accuracy: {war:.3f}'.format(war=war) + '\n')
            f.write('Current Accuracy: {uar:.3f}'.format(uar=uar) + '\n')
    return top1.avg, losses.avg,war,uar


def save_checkpoint(state, is_best, checkpoint_path, best_checkpoint_path):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)

class AverageMeter(object):
    """
    Computes and stores the average and current value:计算并且存储平均值和当前值
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(self.log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """
    注：这里可能需要修改
    Computes the accuracy over the k top predictions for the specified values of k:
    计算指定k值的预测精度
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class RecorderMeter(object):
    """
    Computes and stores the minimum loss value and its epoch index:计算并且存储最小损失值和其对应的epoch索引
    """
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i+1 for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


def plot_confusion_matrix(cm, classes, normalize=False, title='confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=12,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

def computer_uar_war(val_loader, model, best_checkpoint_path, log_confusion_matrix_path, log_txt_path):   
    pre_trained_dict = torch.load(best_checkpoint_path,map_location=f"cuda:{args.gpu}")['state_dict']
    model.load_state_dict(pre_trained_dict)
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for i, (images_face,images_body, target) in enumerate(tqdm.tqdm(val_loader)):
            
            images_face = images_face.to(device)
            images_body = images_body.to(device)
            target = target.to(device)

            output = model(images_face,images_body)           

            predicted = output.argmax(dim=1, keepdim=True)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

            if i == 0:
                all_predicted = predicted
                all_targets = target
            else:
                all_predicted = torch.cat((all_predicted, predicted), 0)
                all_targets = torch.cat((all_targets, target), 0)

    war = 100. * correct / len(val_loader.dataset)
    
    # Compute confusion matrix
    _confusion_matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    np.set_printoptions(precision=4)
    normalized_cm = _confusion_matrix.astype('float') / _confusion_matrix.sum(axis=1)[:, np.newaxis]
    normalized_cm = normalized_cm * 100
    list_diag = np.diag(normalized_cm)
    uar = list_diag.mean()   #对角线的位置除以N,即取平均值
        
    print("Confusion Matrix Diag:", list_diag)
    print("UAR: %0.2f" % uar)
    print("WAR: %0.2f" % war)

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))

    if args.dataset == "FERV39K":
        title_ = "Confusion Matrix on FERV39k"
    elif args.dataset == "DFEW":
        title_ = "Confusion Matrix on DFEW fold "
    elif args.dataset == "MAFW":
        title_ = "Confusion Matrix on MAFW fold "
    elif args.dataset == "AcademicEmotion":
        title_ = "Confusion Matrix on AcademicEmotion"
    elif args.dataset  == "AcademicEmotion_five":
        title_ = "Confusion Matrix on AcademicEmotion"
    elif args.dataset  == "AcademicEmotion_four":
        title_ = "Confusion Matrix on AcademicEmotion"

    plot_confusion_matrix(normalized_cm, classes=class_names, normalize=True, title=title_)
    plt.savefig(os.path.join(log_confusion_matrix_path))
    plt.close()
    
    with open(log_txt_path, 'a') as f:
        f.write('************************' + '\n')
        f.write("Confusion Matrix Diag:" + '\n')
        f.write(str(list_diag.tolist()) + '\n')
        f.write('UAR: {:.2f}'.format(uar) + '\n')        
        f.write('WAR: {:.2f}'.format(war) + '\n')
        f.write('************************' + '\n')
    
    return uar, war


if __name__ == '__main__':
    main()