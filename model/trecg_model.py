import math
import os
import time
from collections import OrderedDict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision
import cv2
import util.utils as util
from util.average_meter import AverageMeter
from . import networks as networks
from .base_model import BaseModel
import torch.nn.functional as F


class TRecgNet(BaseModel):

    def __init__(self, cfg, writer=None):
        super(TRecgNet, self).__init__(cfg)

        util.mkdir(self.save_dir)
        self.phase = cfg.PHASE
        self.trans = not cfg.NO_TRANS
        self.content_model = None
        self.batch_size = cfg.BATCH_SIZE
        self.writer = writer

        # networks
        # self.net = networks.define_netowrks(cfg, device=self.device)
        from lib.sync_bn.modules import BatchNorm2d as SyncBatchNorm

        self.net = networks.define_netowrks(cfg,device=self.device,BatchNorm=SyncBatchNorm)
        self.modules_ori = [self.net.layer0, self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]
        self.modules_new = [self.net.ppm, self.net.cls, self.net.aux]
        self.params_list = []
        for module in self.modules_ori:
            self.params_list.append(dict(params=module.parameters(), lr=cfg.LR))
        for module in self.modules_new:
            self.params_list.append(dict(params=module.parameters(), lr=cfg.LR*10))
        networks.print_network(self.net)

    def build_output_keys(self, trans=True, cls=True):

        out_keys = []

        if trans:
            out_keys.append('gen_img')

        if cls:
            out_keys.append('cls')

        return out_keys

    def _optimize(self, iters):

        self._forward(iters)
        self.optimizer_ED.zero_grad()
        total_loss = self._construct_TRAIN_G_LOSS(iters)
        total_loss.backward()
        self.optimizer_ED.step()

    def set_input(self, data):
        self.source_modal = data['image'].to(self.device)
        self.label = data['label'].to(self.device)
        if self.trans:
            target_modal = data[self.cfg.TARGET_MODAL]

            if isinstance(target_modal, list):
                self.target_modal = list()
                for i, item in enumerate(target_modal):
                    self.target_modal.append(item.to(self.device))
            else:
                # self.target_modal = util.color_label(self.label)
                self.target_modal = target_modal.to(self.device)
        else:
            self.target_modal=None

    def train_parameters(self, cfg):

        assert (self.cfg.LOSS_TYPES)

        if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.EVALUATE:
            # criterion_segmentation = util.CrossEntropyLoss2d(weight=cfg.CLASS_WEIGHTS_TRAIN, ignore_index=cfg.IGNORE_LABEL)
            criterion_segmentation = util.CrossEntropyLoss_PSP(ignore_index=255)

            self.net.set_cls_criterion(criterion_segmentation)

        if 'SEMANTIC' in self.cfg.LOSS_TYPES:
            criterion_content = torch.nn.L1Loss()
            content_model = networks.Content_Model(cfg, criterion_content).to(self.device)
            self.net.set_content_model(content_model)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES:
            criterion_pix2pix = torch.nn.L1Loss()
            self.net.set_pix2pix_criterion(criterion_pix2pix)

        self.set_optimizer(cfg)
        self.set_log_data(cfg)
        self.set_schedulers(cfg)
        self.net = nn.DataParallel(self.net).to(self.device)

        train_total_steps = 0
        train_total_iter = 0

        total_epoch = int(cfg.NITER_TOTAL / math.ceil((self.train_image_num / cfg.BATCH_SIZE)))
        print('total epoch:{0}, total iters:{1}'.format(total_epoch, cfg.NITER_TOTAL))

        for epoch in range(cfg.START_EPOCH, total_epoch + 1):

            if train_total_iter > cfg.NITER_TOTAL:
                break

            self.print_lr()

            self.imgs_all = []
            self.pred_index_all = []
            self.target_index_all = []
            self.fake_image_num = 0

            start_time = time.time()

            data_loader = self.get_dataloader(cfg, epoch)

            self.phase = 'train'
            self.net.train()

            for key in self.loss_meters:
                self.loss_meters[key].reset()
            iters = 0
            self.target_modal=None
            self.val_iou = test_slide(self.val_loader,self.net,self.batch_size)
            print('{epoch}/{total} MIOU: {miou}'.format(epoch=epoch, total=total_epoch, miou=self.val_iou * 100))
            for i, data in enumerate(data_loader):

                self.update_learning_rate(epoch=train_total_iter)
                self.set_input(data)

                train_total_steps += self.batch_size
                train_total_iter += 1
                iters += 1

                self._optimize(train_total_iter)
                # print(i)
                # self._write_loss(phase=self.phase, global_step=train_total_iter)

            print('log_path:', cfg.LOG_PATH)
            print('iters in one epoch:', iters)
            print('gpu_ids:', cfg.GPU_IDS)
            self._write_loss(phase=self.phase, global_step=train_total_iter)
            print('Epoch: {epoch}/{total}'.format(epoch=epoch, total=total_epoch))
            train_errors = self.get_current_errors(current=False)
            print('#' * 10)
            self.print_current_errors(train_errors, epoch)
            print('#' * 10)
            print('Training Time: {0} sec'.format(time.time() - start_time))

            # Validate cls
            # if epoch % 2 == 0 or epoch == total_epoch:
            if train_total_iter > cfg.NITER_TOTAL * 0.7 and epoch % 5 == 0 or epoch == total_epoch or epoch==1 or epoch % 10 == 0:
                if cfg.EVALUATE:
                    if not self.cfg.SLIDE_WINDOWS:
                        self.val_iou = self.validate(train_total_iter)
                        print('{epoch}/{total} MIOU: {miou}'.format(epoch=epoch, total=total_epoch, miou=self.val_iou * 100))
                        self._write_loss(phase=self.phase, global_step=train_total_iter)
                    else:
                        self.val_iou=test_slide(self.val_loader,self.net,self.cfg.BATCH_SIZE)
                        print('{epoch}/{total} MIOU: {miou}'.format(epoch=epoch, total=total_epoch, miou=self.val_iou * 100))
            print('End of iter {0} / {1} \t '
                  'Time Taken: {2} sec'.format(train_total_iter, cfg.NITER_TOTAL, time.time() - start_time))
            print('-' * 80)

    # encoder-decoder branch
    def _forward(self, iters):

        self.gen = None
        self.cls_loss = None

        if self.phase == 'train':

            if 'CLS' not in self.cfg.LOSS_TYPES:
                if_trans = True
                if_cls = False

            elif self.trans and 'CLS' in self.cfg.LOSS_TYPES:
                if_trans = True
                if_cls = True
            else:
                if_trans = False
                if_cls = True
        else:
            if_cls = True
            if_trans = False
            # for time saving
            if iters > self.cfg.NITER_TOTAL - 500 and self.trans:
                if_trans = True

        self.source_modal_show = self.source_modal  # rgb

        out_keys = self.build_output_keys(trans=if_trans, cls=if_cls)
        self.result = self.net(source=self.source_modal, target=self.target_modal, label=self.label, out_keys=out_keys,
                               phase=self.phase)

        if if_cls:
            self.cls = self.result['cls']

        if if_trans:
            if self.cfg.MULTI_MODAL:
                self.gen = [self.result['gen_img_1'], self.result['gen_img_2']]
            else:
                self.gen = self.result['gen_img']

    def _construct_TRAIN_G_LOSS(self, iters):

        loss_total = torch.zeros(1)
        if self.use_gpu:
            loss_total = loss_total.to(self.device)

        if 'CLS' in self.cfg.LOSS_TYPES:
            cls_loss = self.result['loss_cls'].mean() * self.cfg.ALPHA_CLS
            loss_total = loss_total + cls_loss

            cls_loss = round(cls_loss.item(), 4)
            self.loss_meters['TRAIN_CLS_LOSS'].update(cls_loss)
            # self.Train_predicted_label = self.cls.data
            self.Train_predicted_label = self.cls.data.max(1)[1].cpu().numpy()

        # ) content supervised
        if 'SEMANTIC' in self.cfg.LOSS_TYPES:

            decay_coef = 1
            # decay_coef = (iters / self.cfg.NITER_TOTAL)  # small to big
            # decay_coef = max(0, (self.cfg.NITER_TOTAL - iters) / self.cfg.NITER_TOTAL) # big to small
            content_loss = self.result['loss_content'].mean() * self.cfg.ALPHA_CONTENT * decay_coef
            loss_total = loss_total + content_loss

            content_loss = round(content_loss.item(), 4)
            self.loss_meters['TRAIN_SEMANTIC_LOSS'].update(content_loss)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES:

            decay_coef = 1
            # decay_coef = (iters / self.cfg.NITER_TOTAL)  # small to big
            # decay_coef = max(0, (self.cfg.NITER_TOTAL - iters) / self.cfg.NITER_TOTAL) # big to small
            pix2pix_loss = self.result['loss_pix2pix'].mean() * self.cfg.ALPHA_PIX2PIX * decay_coef
            loss_total = loss_total + pix2pix_loss

            pix2pix_loss = round(pix2pix_loss.item(), 4)
            self.loss_meters['TRAIN_PIX2PIX_LOSS'].update(pix2pix_loss)

        return loss_total

    def set_log_data(self, cfg):

        self.loss_meters = defaultdict()
        self.log_keys = [
            'TRAIN_G_LOSS',
            'TRAIN_SEMANTIC_LOSS',  # semantic
            'TRAIN_PIX2PIX_LOSS',
            'TRAIN_CLS_ACC',
            'VAL_CLS_ACC',  # classification
            'TRAIN_CLS_LOSS',
            'VAL_CLS_LOSS',
            'TRAIN_CLS_MEAN_IOU',
            'VAL_CLS_MEAN_IOU',
            'TRAIN_I',
            'TRAIN_U',
            'VAL_I',
            'VAL_U',
            'VAL_TAR',
        ]
        for item in self.log_keys:
            self.loss_meters[item] = AverageMeter()

    def save_checkpoint(self, iter, filename=None):

        if filename is None:
            filename = 'Trans2_{0}_{1}.pth'.format(self.cfg.WHICH_DIRECTION, iter)

        net_state_dict = self.net.state_dict()
        save_state_dict = {}
        for k, v in net_state_dict.items():
            if 'content_model' in k:
                continue
            save_state_dict[k] = v

        state = {
            'iter': iter,
            'state_dict': save_state_dict,
            'optimizer_ED': self.optimizer_ED.state_dict(),
        }

        filepath = os.path.join(self.save_dir, filename)
        torch.save(state, filepath)

    def load_checkpoint(self, net, checkpoint_path, checkpoint, optimizer=None, data_para=True):

        keep_fc = not self.cfg.NO_FC

        if os.path.isfile(checkpoint_path):

            # load from pix2pix net_G, no cls weights, selected update
            state_dict = net.state_dict()
            state_checkpoint = checkpoint['state_dict']
            if data_para:
                new_state_dict = OrderedDict()
                for k, v in state_checkpoint.items():
                    name = k[7:]
                    new_state_dict[name] = v
                state_checkpoint = new_state_dict

            if keep_fc:
                pretrained_G = {k: v for k, v in state_checkpoint.items() if k in state_dict}
            else:
                pretrained_G = {k: v for k, v in state_checkpoint.items() if k in state_dict and 'fc' not in k}

            state_dict.update(pretrained_G)
            net.load_state_dict(state_dict)

            if self.phase == 'train' and not self.cfg.INIT_EPOCH:
                optimizer.load_state_dict(checkpoint['optimizer_ED'])

            # print("=> loaded checkpoint '{}' (iter {})"
            #       .format(checkpoint_path, checkpoint['iter']))
        else:
            print("=> !!! No checkpoint found at '{}'".format(self.cfg.RESUME))
            return

    def set_optimizer(self, cfg):

        self.optimizers = []
        # self.optimizer_ED = torch.optim.Adam([{'params': self.net.fc.parameters(), 'lr': cfg.LR}], lr=cfg.LR / 10, betas=(0.5, 0.999))

        # self.optimizer_ED = torch.optim.Adam(self.net.parameters(), lr=cfg.LR, betas=(0.5, 0.999))
        self.optimizer_ED = torch.optim.SGD(self.params_list,lr=cfg.LR, momentum=cfg.MOMENTUM,weight_decay=cfg.WEIGHT_DECAY)
        print('optimizer: ', self.optimizer_ED)
        self.optimizers.append(self.optimizer_ED)


    def validate(self, iters):

        self.phase = 'test'

        # switch to evaluate mode
        self.net.eval()

        self.imgs_all = []
        self.pred_index_all = []
        self.target_index_all = []

        inputs_all, gts_all, predictions_all = [], [], []
        with torch.no_grad():
            print('# Cls val images num = {0}'.format(self.val_image_num))
            # batch_index = int(self.val_image_num / cfg.BATCH_SIZE)
            # random_id = random.randint(0, batch_index)
            # confusion_matrix = np.zeros((37,37))
            for i, data in enumerate(self.val_loader):
                # self.set_input(data, self.cfg.DATA_TYPE)
                self.source_modal = data['image'].to(self.device)
                # self.target_modal=data['depth'].cuda()
                self.label = data['label'].to(self.device)

                self._forward(iters)
                cls_loss = self.result['loss_cls'].mean() * self.cfg.ALPHA_CLS
                self.loss_meters['VAL_CLS_LOSS'].update(cls_loss)
                self.val_predicted_label = self.cls.data.max(1)[1].cpu().numpy()
                # self.val_predicted_label = self.cls.data.max(1)[1]


                # gts_all.append(self.label.data.cpu().numpy())
                # predictions_all.append(self.val_predicted_label)
                # print(i)

                # self.val_predicted_label = self.cls.data
                # _, pred = torch.max(self.cls.data, dim=1)
                # acc, pix = util.accuracy(pred + 1, self.label.long())
                intersection,union = util.intersectionAndUnion(self.val_predicted_label, self.label,self.cfg.NUM_CLASSES)
                # self.loss_meters['VAL_CLS_ACC'].update(acc, pix, acc_Flag=True)
                # intersection,union,target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                self.loss_meters['VAL_I'].update(intersection)
                self.loss_meters['VAL_U'].update(union)
                # self.loss_meters['VAL_TAR'].update(target)


        # gts_all = np.concatenate(gts_all)
        # predictions_all = np.concatenate(predictions_all)

        # acc, acc_cls, mean_iu, fwavacc = util.evaluate(predictions_all, gts_all, self.cfg.NUM_CLASSES)
        mean_iu=self.loss_meters['VAL_I'].sum/ (self.loss_meters['VAL_U'].sum + 1e-10)
        return np.mean(mean_iu)

    def _write_loss(self, phase, global_step):

        loss_types = self.cfg.LOSS_TYPES

        self.label_show = self.label.data.cpu().numpy()
        self.source_modal_show = self.source_modal
        self.target_modal_show = self.target_modal

        if phase == 'train':

            self.writer.add_scalar('Seg/LR', self.optimizer_ED.param_groups[0]['lr'], global_step=global_step)

            if 'CLS' in loss_types:
                self.writer.add_scalar('Seg/TRAIN_CLS_LOSS', self.loss_meters['TRAIN_CLS_LOSS'].avg,
                                       global_step=global_step)
                # self.writer.add_scalar('TRAIN_CLS_ACC', self.loss_meters['TRAIN_CLS_ACC'].avg*100.0,
                #                        global_step=global_step)
                # self.writer.add_scalar('TRAIN_CLS_MEAN_IOU', float(self.train_iou.mean())*100.0,
                #                        global_step=global_step)

            if self.trans:

                if 'SEMANTIC' in self.cfg.LOSS_TYPES:
                    self.writer.add_scalar('Seg/TRAIN_SEMANTIC_LOSS', self.loss_meters['TRAIN_SEMANTIC_LOSS'].avg,
                                           global_step=global_step)
                if 'PIX2PIX' in self.cfg.LOSS_TYPES:
                    self.writer.add_scalar('Seg/TRAIN_PIX2PIX_LOSS', self.loss_meters['TRAIN_PIX2PIX_LOSS'].avg,
                                           global_step=global_step)

                if isinstance(self.target_modal, list):
                    for i, (gen, target) in enumerate(zip(self.gen, self.target_modal)):
                        self.writer.add_image('Seg/2_Train_Gen_' + str(self.cfg.FINE_SIZE / pow(2, i)),
                                              torchvision.utils.make_grid(gen[:6].clone().cpu().data, 3,
                                                                          normalize=True),
                                              global_step=global_step)
                        self.writer.add_image('Seg/3_Train_Target_' + str(self.cfg.FINE_SIZE / pow(2, i)),
                                              torchvision.utils.make_grid(target[:6].clone().cpu().data, 3,
                                                                          normalize=True),
                                              global_step=global_step)
                else:
                    self.writer.add_image('Seg/Train_target',
                                          torchvision.utils.make_grid(self.target_modal_show[:6].clone().cpu().data, 3,
                                                                      normalize=True), global_step=global_step)
                    self.writer.add_image('Seg/Train_gen',
                                          torchvision.utils.make_grid(self.gen.data[:6].clone().cpu().data, 3,
                                                                      normalize=True), global_step=global_step)

            self.writer.add_image('Seg/Train_image',
                                  torchvision.utils.make_grid(self.source_modal_show[:6].clone().cpu().data, 3,
                                                              normalize=True), global_step=global_step)
            if 'CLS' in loss_types:
                self.writer.add_image('Seg/Train_predicted',
                                      torchvision.utils.make_grid(
                                          torch.from_numpy(util.color_label(self.Train_predicted_label[:6], ignore=self.cfg.IGNORE_LABEL)), 3,
                                          normalize=True, range=(0, 255)), global_step=global_step)
                # torchvision.utils.make_grid(util.color_label(torch.max(self.Train_predicted_label[:6], 1)[1]+1), 3, normalize=False,range=(0, 255)), global_step=global_step)
                self.writer.add_image('Seg/Train_label',
                                      torchvision.utils.make_grid(
                                          torch.from_numpy(util.color_label(self.label_show[:6], ignore=self.cfg.IGNORE_LABEL)), 3, normalize=True,
                                          range=(0, 255)), global_step=global_step)

        if phase == 'test':
            self.writer.add_image('Seg/Val_image',
                                  torchvision.utils.make_grid(self.source_modal_show[:6].clone().cpu().data, 3,
                                                              normalize=True), global_step=global_step)

            self.writer.add_image('Seg/Val_predicted',
                                  torchvision.utils.make_grid(
                                      torch.from_numpy(util.color_label(self.val_predicted_label[:6], ignore=self.cfg.IGNORE_LABEL)), 3,
                                      normalize=True, range=(0, 255)), global_step=global_step)
            # torchvision.utils.make_grid(util.color_label(torch.max(self.val_predicted_label[:3], 1)[1]+1), 3, normalize=False,range=(0, 255)), global_step=global_step)
            self.writer.add_image('Seg/Val_label',
                                  torchvision.utils.make_grid(torch.from_numpy(util.color_label(self.label_show[:6], ignore=self.cfg.IGNORE_LABEL)),
                                                              3, normalize=True, range=(0, 255)),
                                  global_step=global_step)

            self.writer.add_scalar('Seg/VAL_CLS_LOSS', self.loss_meters['VAL_CLS_LOSS'].avg,
                                   global_step=global_step)
            # self.writer.add_scalar('Seg/VAL_CLS_ACC', self.loss_meters['VAL_CLS_ACC'].avg*100.0,
            #                        global_step=global_step)
            self.writer.add_scalar('Seg/VAL_CLS_MEAN_IOU', float(self.val_iou.mean()) * 100.0,
                                   global_step=global_step)
            # self.writer.add_scalar('Seg/VAL_CLS_MEAN_IOU', float(self.val_iou.mean())*100.0,
            #                        global_step=global_step)


def get_confusion_matrix(gt_label, pred_label, class_num=37):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix
def net_process(model, image):
    input = torch.from_numpy(image).float()
    with torch.no_grad():
        output = model(source=input,phase='test')
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)

    output = output.data.cpu().numpy()
    return output

def scale_process(model, image, batchsize,classes, crop_h, crop_w, h, w, stride_rate=2/3):
    _,_,new_h, new_w = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = np.zeros((batchsize,classes,new_h, new_w), dtype=float)
    count_crop = np.zeros((batchsize,1,new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[:,:,s_h:e_h, s_w:e_w].copy()
            count_crop[:,:,s_h:e_h, s_w:e_w] += 1
            prediction_crop[:,:,s_h:e_h, s_w:e_w] += net_process(model, image_crop)
    prediction_crop /=count_crop
    return prediction_crop


def test_slide(test_loader, model,batch_size,classes=19, base_size=2048, crop_h=713, crop_w=713, scales=[1.0]):
    print('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    model.eval()
    for i, data in enumerate(test_loader):
        input=data['image']
        target=data['label']
        input = input.numpy()
        _,_,h, w= input.shape
        prediction = np.zeros((batch_size,classes,h, w), dtype=float)
        prediction = scale_process(model, input,batch_size, classes, crop_h, crop_w, h, w)
        prediction = np.argmax(prediction, axis=1)
        pred = np.uint8(prediction)
        target = np.uint8(target)
        intersection, union, target = util.intersectionAndUnion_SW(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        print("rate:{},acc:{}".format(i,accuracy))
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    print('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    # print('using time:'(time.time() - end))
    print('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return mIoU

