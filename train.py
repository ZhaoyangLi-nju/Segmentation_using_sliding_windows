import os
import random
from functools import reduce

import torch
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader  # new add

import util.utils as util
from config.default_config import DefaultConfig
from config.resnet18_sunrgbd_config import RESNET18_SUNRGBD_CONFIG
from data import segmentation_dataset_cv2 as segmentation_dataset
from model.trecg_model import TRecgNet
from model.trecg_model_multimodal import TRecgNet_MULTIMODAL
import torch.backends.cudnn as cudnn
import sys
from datetime import datetime

# alpha = sys.argv[1]
cfg = DefaultConfig()
args = {
    'resnet18': RESNET18_SUNRGBD_CONFIG().args(),
}
# args for different backbones
cfg.parse(args['resnet18'])

if len(sys.argv) > 1:
    device_ids = torch.cuda.device_count()
    print('device_ids:', device_ids)

    arg_model, arg_target, arg_alpha, arg_loss, arg_task = sys.argv[1:]
    # print(arg_model, arg_norm, arg_alpha)

    cfg.MODEL = arg_model
    cfg.TARGET_MODAL = arg_target
    cfg.ALPHA_CONTENT = float(arg_alpha)
    cfg.LOG_PATH = os.path.join('/home/lzy/summary/', cfg.MODEL, cfg.PRETRAINED,
                                ''.join([arg_task, '_2', cfg.TARGET_MODAL, '_alpha_', arg_alpha, '_', arg_loss,
                                         '_gpus-', str(device_ids)]), datetime.now().strftime('%b%d_%H-%M-%S'))
    cfg.LOSS_TYPES = [arg_loss, 'CLS']
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_IDS

# Setting random seed
if cfg.MANUAL_SEED is None:
    cfg.MANUAL_SEED = random.randint(1, 10000)
random.seed(cfg.MANUAL_SEED)
torch.manual_seed(cfg.MANUAL_SEED)

torch.backends.cudnn.benchmark = True

project_name = reduce(lambda x, y: str(x) + '/' + str(y), os.path.realpath(__file__).split(os.sep)[:-1])
util.mkdir('logs')

# resize_size = 256
# crop_size = 224
# image_h,image_w=416,544
value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]
train_transforms = list()
ms_targets = []
train_transforms.append(segmentation_dataset.Resize(cfg.LOAD_SIZE_cityscapes))
train_transforms.append(segmentation_dataset.RandomScale(cfg.RANDOM_SCALE_SIZE))#
# train_transforms.append(segmentation_dataset.RandomRotate())#
# train_transforms.append(segmentation_dataset.RandomGaussianBlur())
# train_transforms.append(segmentation_dataset.ColorJitter(brightness = 0.5,contrast = 0.5,saturation = 0.5))
# train_transforms.append(segmentation_dataset.RandomCrop(cfg.FINE_SIZE_cityscapes))#
train_transforms.append(segmentation_dataset.Crop(cfg.FINE_SIZE_cityscapes,crop_type='rand'))#

train_transforms.append(segmentation_dataset.RandomHorizontalFlip())

# if cfg.MULTI_SCALE:
#     for item in cfg.MULTI_TARGETS:
#         ms_targets.append(item)
#     train_transforms.append(segmentation_dataset.MultiScale(size=(cfg.FINE_SIZE, cfg.FINE_SIZE),
#                                                             scale_times=cfg.MULTI_SCALE_NUM, ms_targets=ms_targets))
train_transforms.append(segmentation_dataset.ToTensor(ms_targets=ms_targets))
train_transforms.append(
    segmentation_dataset.Normalize(mean=mean,std=std, ms_targets=ms_targets))

val_transforms = list()
# val_transforms.append(segmentation_dataset.Resize(cfg.LOAD_SIZE_cityscapes))
# val_transforms.append(segmentation_dataset.Crop((713,713), crop_type='center'))
val_transforms.append(segmentation_dataset.ToTensor())
val_transforms.append(segmentation_dataset.Normalize(mean=mean,std=std))

train_data = segmentation_dataset.CityScapes(cfg=cfg, transform=transforms.Compose(train_transforms), phase_train=True,
                                          data_dir='/home/lzy/cityscapes')
val_data = segmentation_dataset.CityScapes(cfg=cfg, transform=transforms.Compose(val_transforms), phase_train=False,
                                        data_dir='/home/lzy/cityscapes')

train_loader = DataLoader(train_data, batch_size=cfg.BATCH_SIZE, shuffle=True,
                          num_workers=cfg.WORKERS, pin_memory=True,drop_last=True)
val_loader = DataLoader(val_data, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=cfg.WORKERS, pin_memory=True)
# val_loader=None
unlabeled_loader = None
num_train = len(train_data)
num_val = len(val_data)

cfg.CLASS_WEIGHTS_TRAIN = train_data.class_weights
cfg.IGNORE_LABEL = train_data.ignore_label

# shell script to run
print('LOSS_TYPES:', cfg.LOSS_TYPES)
writer = SummaryWriter(log_dir=cfg.LOG_PATH)  # tensorboard

if cfg.MULTI_MODAL:
    model = TRecgNet_MULTIMODAL(cfg, writer=writer)
else:
    model = TRecgNet(cfg, writer=writer)
model.set_data_loader(train_loader, val_loader, unlabeled_loader, num_train, num_val)

def train():
    if cfg.RESUME:
        checkpoint_path = os.path.join(cfg.CHECKPOINTS_DIR, cfg.RESUME_PATH)
        checkpoint = torch.load(checkpoint_path)
        # load_epoch = checkpoint['epoch']
        model.load_checkpoint(model.net, checkpoint_path, checkpoint, data_para=True)
        # cfg.START_EPOCH = load_epoch

        if cfg.INIT_EPOCH:
            # just load pretrained parameters
            print('load checkpoint from another source')
            cfg.START_EPOCH = 1

    print('>>> task path is {0}'.format(project_name))

    # train
    model.train_parameters(cfg)

    print('save model ...')
    model_filename = '{0}_{1}_{2}.pth'.format(cfg.MODEL, cfg.WHICH_DIRECTION, cfg.NITER_TOTAL)
    model.save_checkpoint(cfg.NITER_TOTAL, model_filename)

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    train()
