#### 注意提示词位置，提示词，和模型权重都要对应才可以
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
from models.Generate_Model_v5 import GenerateModel

parser = argparse.ArgumentParser()
# 基础设置(数据集、显卡、workers)
parser.add_argument('--dataset', type=str,default='AcademicEmotion_five')
parser.add_argument('--gpu', type=int,default = 1)
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

parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=4)
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
parser.add_argument('--seed', type=int,default=1)  

# load_and_tune_prompt_learner(是否微调提示学习)
parser.add_argument('--load_and_tune_prompt_learner', type=str, default='True')
args = parser.parse_args()

# 选择设备
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
args.device = device


# 结果存储设置
now = datetime.datetime.now()
time_str = now.strftime("-[%m-%d]-[%H:%M]")
args.name = args.exper_name + str(time_str) 

if args.resume == 'True':
    args.name = args.resume_path
args.output_path = "outputs/" + args.name + "/"

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)


if args.dataset == "AcademicEmotion_five":
    class_names_5 = [
    'Neutrality',
    'Enjoyment',
    'Confusion',
    'Fatigue',
    'Distraction.'
    ]
    class_names = class_names_5
    class_names_with_context = class_names_with_context_5
    class_descriptor = class_descriptor_5



def main():
    if args.dataset == "AcademicEmotion_five":
        print("*********** AcademicEmotion_six Dataset ***********")
        log_txt_path = args.output_path + 'log.txt'
        log_curve_path = args.output_path  + 'log.png'
        log_confusion_matrix_path = args.output_path + 'cnconfusion_matrix.png'
        checkpoint_path = args.output_path + 'model.pth'
        best_checkpoint_path = "/media/D/zlm/code/CLIP_CAER/pritrained_checkpoint/model_best.pth"
        # 这里可以换位比较固定的数据集标签路径，而不是一个代码一个
        test_annotation_file_path = "/media/D/zlm/code/CLIP_CAER/different_context/conputer_and_mouse.txt"
        
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
    # 加载预训练权重

    model = model.to(device)

    # print params   
    print('************************')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('************************')
    

    test_data = test_data_loader(list_file=test_annotation_file_path,
                                 num_segments=16,
                                 duration=1,
                                 image_size=224,
                                 over_sample= False,
                                 )
    
    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)
    uar, war = computer_uar_war(val_loader, model, best_checkpoint_path, log_confusion_matrix_path, log_txt_path)
    return uar, war


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
    # 将pre_trained_dict保存
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

    labels = [0, 1, 2, 3, 4]
    # Compute confusion matrix
    _confusion_matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy(),labels=labels)
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
        title_ = "Confusion Matrix on REAR"
    elif args.dataset  == "AcademicEmotion_four":
        title_ = "Confusion Matrix on REAR"

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