import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import copy

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_images(dir, extensions):
    images = []
    dir = os.path.expanduser(dir)
    image_names = [d for d in os.listdir(dir)]
    for image_name in image_names:
        if has_file_allowed_extension(image_name, extensions):
            file = os.path.join(dir, image_name)
            images.append(file)
    return images


#Checks if a file is an allowed extension.
def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res

def mean_acc(target_indice, pred_indice, num_classes, classes=None):
    assert(num_classes == len(classes))
    acc = 0.
    print('{0} Class Acc Report {1}'.format('#' * 10, '#' * 10))
    for i in range(num_classes):
        idx = np.where(target_indice == i)[0]
        # acc = acc + accuracy_score(target_indice[idx], pred_indice[idx])
        class_correct = accuracy_score(target_indice[idx], pred_indice[idx])
        acc += class_correct
        print('acc {0}: {1:.3f}'.format(classes[i], class_correct * 100))

        # class report
        # y_tpye, y_true, y_pred = _check_targets(target_indice[idx], pred_indice[idx])
        # score = y_true == y_pred
        # wrong_index = np.where(score == False)[0]
        # for j in idx[wrong_index]:
        #     print("Wrong for class [%s]: predicted as: <%s>, image_id--<%s>" %
        #           (int_to_class[i], int_to_class[pred[j]], image_paths[j]))
        #
        # print("[class] %s accuracy is %.3f" % (int_to_class[i], class_correct))
    print('#' * 30)
    return (acc / num_classes) * 100

def process_output(output):
    # Computes the result and argmax index
    pred, index = output.topk(1, 1, largest=True)

    return pred.cpu().float().numpy().flatten(), index.cpu().numpy().flatten()

# class CrossEntropyLoss2d(nn.Module):
#     def __init__(self, weight=med_frq):
#         super(CrossEntropyLoss2d, self).__init__()
#         self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(),
#                                            size_average=False, reduce=False)
#     def forward(self, inputs_scales, targets_scales):
#         # losses = []
#         # for inputs, targets in zip(inputs_scales, targets_scales):
#         mask = targets_scales > 0
#         targets_m = targets_scales.copy()
#         targets_m[mask] -= 1
#         loss_all = self.ce_loss(inputs_scales, targets_m.long())
#         total_loss=(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
#         # total_loss = sum(losses)
#         return total_loss
class CrossEntropyLoss_PSP(nn.Module):
    def __init__(self,ignore_index=255):
        super(CrossEntropyLoss_PSP, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    def forward(self, inputs_scales, targets_scales):
        return self.ce_loss(inputs_scales, targets_scales.long())
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        if weight:
            weight = torch.FloatTensor(weight).cuda()
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)


# def intersectionAndUnionSUNRGBD(imPred, imLab, numClass):
#     imPred=imPred.cpu().numpy()
#     imLab=imLab.cpu().numpy()
#     imPred = np.asarray(imPred,dtype=np.int).copy()
#     imLab = np.asarray(imLab,dtype=np.int).copy()

#     imPred +=1 
#     imPred = imPred * ((imLab>0)&(imLab!=255))
#     intersection = imPred * (imPred == imLab)
#     (area_intersection, _) = np.histogram(
#         intersection, bins=numClass, range=(1, numClass+1))
#     # Compute area union:
#     (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass+1))
#     (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass+1))
#     area_union = area_pred + area_lab - area_intersection
#     return (area_intersection, area_union)
def intersectionAndUnion(imPred, imLab, numClass):
    # imPred=imPred.cpu().numpy()
    imLab=imLab.cpu().numpy()
    imPred = np.asarray(imPred,dtype=np.int).copy()
    imLab = np.asarray(imLab,dtype=np.int).copy()

    # imPred +=1 
    imPred = imPred * (imLab!=255)
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(0, numClass))
    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(0, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(0, numClass))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union)
def intersectionAndUnion_SW(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target
# def intersectionAndUnion(output, target, K, ignore_index=255):
#     # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
#     output = output.view(-1)
#     target = target.view(-1)
#     output[target == ignore_index] = ignore_index
#     intersection = output[output == target]
#     # https://github.com/pytorch/pytorch/issues/1382
#     area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
#     area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
#     area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
#     area_union = area_output + area_target - area_intersection
#     return area_intersection.cuda(), area_union.cuda(), area_target.cuda()

def accuracy(preds, label):
    valid = (label > 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / float(valid_sum + 1e-10)
    return acc, valid_sum

# def color_label(label):
#     label = label.clone().cpu().data.numpy()
#     colored_label = np.vectorize(lambda x: label_colours[int(x)])
#
#     colored = np.asarray(colored_label(label)).astype(np.float32)
#     colored = colored.squeeze()
#
#     try:
#         return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
#     except ValueError:
#         return torch.from_numpy(colored[np.newaxis, ...])

def color_label_np(label, ignore=None):
    colored_label = np.vectorize(lambda x: label_colours[-1] if x == ignore else label_colours[int(x)])
    colored = np.asarray(colored_label(label)).astype(np.float32)
    # colored = colored.squeeze()

    try:
        return colored.transpose([1, 2, 0])
    except ValueError:
        return colored[np.newaxis, ...]


def color_label(label, ignore=None):
    # label = label.data.cpu().numpy()
    colored_label = np.vectorize(lambda x: label_colours[-1] if x == ignore else label_colours[int(x)])
    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()

    try:
        return colored.transpose([1, 0, 2, 3])
    except ValueError:
        return colored[np.newaxis, ...]


label_colours = [
                 (148, 65, 137), (255, 116, 69), (86, 156, 137),
                 (202, 179, 158), (155, 99, 235), (161, 107, 108),
                 (133, 160, 103), (76, 152, 126), (84, 62, 35),
                 (44, 80, 130), (31, 184, 157), (101, 144, 77),
                 (23, 197, 62), (141, 168, 145), (142, 151, 136),
                 (115, 201, 77), (100, 216, 255), (57, 156, 36),
                 (88, 108, 129), (105, 129, 112), (42, 137, 126),
                 (155, 108, 249), (166, 148, 143), (81, 91, 87),
                 (100, 124, 51), (73, 131, 121), (157, 210, 220),
                 (134, 181, 60), (221, 223, 147), (123, 108, 131),
                 (161, 66, 179), (163, 221, 160), (31, 146, 98),
                 (99, 121, 30), (49, 89, 240), (116, 108, 9),
                 (139, 110, 246), (0, 0, 0)]  # list(-1) for 255

# palette = [148, 65, 137, 255, 116, 69, 86, 156, 137,
#                  202, 179, 158, 155, 99, 235, 161, 107, 108,
#                  133, 160, 103, 76, 152, 126, 84, 62, 35,
#                  44, 80, 130, 31, 184, 157, 101, 144, 77,
#                  23, 197, 62, 141, 168, 145, 142, 151, 136,
#                  115, 201, 77, 100, 216, 255,57, 156, 36,
#                  88, 108, 129, 105, 129, 112,42, 137, 126,
#                  155, 108, 249, 166, 148, 143, 81, 91, 87,
#                  100, 124, 51, 73, 131, 121, 157, 210, 220,
#                  134, 181, 60, 221, 223, 147, 123, 108, 131,
#                  161, 66, 179, 163, 221, 160, 31, 146, 98,
#                  99, 121, 30, 49, 89, 240, 116, 108, 9,
#                  139, 110, 246]
#
# zero_pad = 256 * 3 - len(palette)
# for i in range(zero_pad):
#     palette.append(0)

# def colorize_mask(mask):
#     # mask: numpy array of the mask
#     # mask = mask.data.cpu().numpy()
#     new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
#     new_mask.putpalette(palette)
#     return np.array(new_mask)



def evaluate(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist