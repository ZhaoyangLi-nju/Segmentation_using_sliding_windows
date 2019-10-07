import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from . import resnet  as models
from torch.nn import init
from torchvision.models.resnet import resnet18


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def fix_grad(net):
    print(net.__class__.__name__)

    def fix_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('BatchNorm2d') != -1:
            m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.requires_grad = False

    net.apply(fix_func)


def unfix_grad(net):
    def fix_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('BatchNorm2d') != -1 or classname.find('Linear') != -1:
            m.weight.requires_grad = True
            if m.bias is not None:
                m.bias.requires_grad = True

    net.apply(fix_func)


def define_netowrks(cfg, device=None,BatchNorm=None):

    if 'resnet' in cfg.ARCH:
        if cfg.MULTI_SCALE:
            model = FCN_Conc_Multiscale(cfg, device=device)
        elif cfg.MULTI_MODAL:
            model = FCN_Conc_MultiModalTarget(cfg, device=device)
        else:
            if cfg.MODEL == 'FCN':
                model = FCN_Conc(cfg, device=device)
            elif cfg.MODEL == 'UNET_256':
                model = UNet_Share_256(cfg, device=device)
            elif cfg.MODEL == 'UNET_128':
                model = UNet_Share_128(cfg, device=device)
            elif cfg.MODEL == 'UNET_64':
                model = UNet_Share_64(cfg, device=device)
            elif cfg.MODEL == 'UNET_LONG':
                model = UNet_Long(cfg, device=device)
            elif cfg.MODEL == "PSP":
                model = PSP_Conc_beachmark(cfg=cfg,device=device,SyncBatchNorm=BatchNorm)
    return model


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv_norm_relu(dim_in, dim_out, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d,
                   use_leakyRelu=False, use_bias=False, is_Sequential=True):
    if use_leakyRelu:
        act = nn.LeakyReLU(0.2, True)
    else:
        act = nn.ReLU(True)

    if is_Sequential:
        result = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=use_bias),
            norm(dim_out, affine=True),
            act
        )
        return result
    return [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            norm(dim_out, affine=True),
            act]


def expand_Conv(module, in_channels):
    def expand_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m.in_channels = in_channels
            m.out_channels = m.out_channels
            mean_weight = torch.mean(m.weight, dim=1, keepdim=True)
            m.weight.data = mean_weight.repeat(1, in_channels, 1, 1).data

    module.apply(expand_func)
class Conc_Up_Residual_bottleneck(nn.Module):

    def __init__(self, dim_in, dim_out, stride=1, norm=nn.BatchNorm2d, conc_feat=True):
        super(Conc_Up_Residual_bottleneck, self).__init__()

        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1,
                      padding=0, bias=False),
            norm(dim_out))
        self.smooth = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1,
                      padding=1, bias=False)

        if conc_feat:
            dim_in = dim_out * 2

        dim_med = int(dim_out / 2)
        self.conv1 = nn.Conv2d(dim_in, dim_med, kernel_size=1, stride=stride,
                      padding=0, bias=False)
        self.norm1 = norm(dim_med)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_med, dim_med)
        self.norm2 = norm(dim_med)
        self.conv3 = nn.Conv2d(dim_med, dim_out, kernel_size=1, stride=1,
                      padding=0, bias=False)
        self.norm3 = norm(dim_out)

    def forward(self, x, y=None):
        if y is not None:
            x = F.interpolate(x, y.size()[2:],mode='bilinear', align_corners=True)
        else:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.smooth(x)
        residual = self.residual_conv(x)

        if y is not None:
            x = torch.cat((x, y), 1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        x += residual

        return self.relu(x)
class Conc_Up_Residual(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d, conc_feat=True):
        super(Conc_Up_Residual, self).__init__()

        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1,
                      padding=0, bias=False),
            norm(dim_out))
        self.smooth = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1,
                      padding=1, bias=False)

        if conc_feat:
            dim_in = dim_out * 2
            kernel_size, padding = 1, 0
        else:
            kernel_size, padding = 3, 1

        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False)
        self.norm1 = norm(dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_out, dim_out)
        self.norm2 = norm(dim_out)

    def forward(self, x, y=None):

        if y is not None:
            x = F.interpolate(x, y.size()[2:],mode='bilinear', align_corners=True)
        else:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.smooth(x)
        residual = self.residual_conv(x)
        if y is not None:
            x = torch.cat((x, y), 1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)

        x += residual

        return self.relu(x)
#########################################

##############################################################################
# Translate to recognize
##############################################################################
class Content_Model(nn.Module):

    def __init__(self, cfg, criterion=None):
        super(Content_Model, self).__init__()
        self.cfg = cfg
        self.criterion = criterion
        self.net = cfg.WHICH_CONTENT_NET

        if 'resnet' in self.net:
            from .pretrained_resnet import ResNet
            self.model = ResNet(self.net, cfg) 

        fix_grad(self.model)
        # print_network(self)

    def forward(self, x, target, in_channel=3, layers=None):

        # important when set content_model as the attr of trecg_net
        self.model.eval()

        layers = layers
        if layers is None or not layers:
            layers = self.cfg.CONTENT_LAYERS.split(',')

        input_features = self.model((x + 1) / 2, layers)
        target_targets = self.model((target + 1) / 2, layers)
        len_layers = len(layers)
        loss_fns = [self.criterion] * len_layers
        alpha = [1] * len_layers

        content_losses = [alpha[i] * loss_fns[i](gen_content, target_targets[i])
                          for i, gen_content in enumerate(input_features)]
        loss = sum(content_losses)
        return loss

class FCN_Conc(nn.Module):

    def __init__(self, cfg, device=None):
        super(FCN_Conc, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        self.head = _FCNHead(512, num_classes, nn.BatchNorm2d)
        # self.head = nn.Conv2d(512, num_classes, 1)

        if self.trans:
            self.build_upsample_content_layers(dims)

        self.score_256 = nn.Sequential(
            nn.Conv2d(dims[3], num_classes, 1)
        )

        self.score_128 = nn.Sequential(
            nn.Conv2d(dims[2], num_classes, 1)
        )
        self.score_64 = nn.Sequential(
            nn.Conv2d(dims[1], num_classes, 1)
        )

        if pretrained:
            init_weights(self.head, 'normal')

            if self.trans:
                init_weights(self.up1, 'normal')
                init_weights(self.up2, 'normal')
                init_weights(self.up3, 'normal')
                init_weights(self.up4, 'normal')

            init_weights(self.head, 'normal')
            init_weights(self.score_64, 'normal')
            init_weights(self.score_128, 'normal')
            init_weights(self.score_256, 'normal')

        else:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def build_upsample_content_layers(self, dims):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        self.up1 = Conc_Up_Residual(dims[4], dims[3], norm=norm)
        self.up2 = Conc_Up_Residual(dims[3], dims[2], norm=norm)
        self.up3 = Conc_Up_Residual(dims[2], dims[1], norm=norm)
        self.up4 = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)

        self.up_image_content = nn.Sequential(
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
                return_losses=True):
        result = {}

        layer_0 = self.relu(self.bn1(self.conv1(source)))
        if not self.trans:
            layer_0 = self.maxpool(layer_0)
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        if self.trans:
            # content model branch

            up1 = self.up1(layer_4, layer_3)
            up2 = self.up2(up1, layer_2)
            up3 = self.up3(up2, layer_1)
            up4 = self.up4(up3)

            result['gen_img'] = self.up_image_content(up4)

        # segmentation branch
        score_512 = self.head(layer_4)

        score_256 = None
        score_128 = None
        score_64 = None
        if self.cfg.WHICH_SCORE == 'main':
            score_256 = self.score_256(layer_3)
            score_128 = self.score_128(layer_2)
            score_64 = self.score_64(layer_1)
        elif self.cfg.WHICH_SCORE == 'up':
            score_256 = self.score_256(up1)
            score_128 = self.score_128(up2)
            score_64 = self.score_64(up3)
        elif self.cfg.WHICH_SCORE == 'both':
            score_256 = self.score_256(up1 + layer_3)
            score_128 = self.score_128(up2 + layer_2)
            score_64 = self.score_64(up3 + layer_1)

        score = F.interpolate(score_512, score_256.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_256
        score = F.interpolate(score, score_128.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_128
        score = F.interpolate(score, score_64.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_64

        result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

        if 'SEMANTIC' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

        if 'CLS' in self.cfg.LOSS_TYPES:
            result['loss_cls'] = self.cls_criterion(result['cls'], label)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)

        return result


class FCN_Conc_MultiModalTarget(nn.Module):

    def __init__(self, cfg, device=None):
        super(FCN_Conc_MultiModalTarget, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        self.head = _FCNHead(512, num_classes, nn.BatchNorm2d)
        # self.head = nn.Conv2d(512, num_classes, 1)

        if self.trans:
            self.build_upsample_content_layers(dims)

        self.score_256 = nn.Sequential(
            nn.Conv2d(dims[3] * 2, num_classes, 1)
        )

        self.score_128 = nn.Sequential(
            nn.Conv2d(dims[2] * 2, num_classes, 1)
        )
        self.score_64 = nn.Sequential(
            nn.Conv2d(dims[1] * 2, num_classes, 1)
        )

            # self.score_256_depth = nn.Conv2d(256, num_classes, 1)
            # self.score_128_depth = nn.Conv2d(128, num_classes, 1)
            # self.score_64_depth = nn.Conv2d(64, num_classes, 1)
            #
            # self.score_256_seg = nn.Conv2d(256, num_classes, 1)
            # self.score_128_seg = nn.Conv2d(128, num_classes, 1)
            # self.score_64_seg = nn.Conv2d(64, num_classes, 1)

        # self.avgpool = nn.AvgPool2d(self.avg_pool_size, 1)
        # self.fc = nn.Linear(fc_input_nc, cfg.NUM_CLASSES)

        if pretrained:
            init_weights(self.head, 'normal')

            if self.trans:
                init_weights(self.up1_depth, 'normal')
                init_weights(self.up2_depth, 'normal')
                init_weights(self.up3_depth, 'normal')
                init_weights(self.up4_depth, 'normal')
                init_weights(self.up1_seg, 'normal')
                init_weights(self.up2_seg, 'normal')
                init_weights(self.up3_seg, 'normal')
                init_weights(self.up4_seg, 'normal')

                # init_weights(self.score_64_depth, 'normal')
                # init_weights(self.score_128_depth, 'normal')
                # init_weights(self.score_256_depth, 'normal')
                # init_weights(self.score_64_seg, 'normal')
                # init_weights(self.score_128_seg, 'normal')
                # init_weights(self.score_256_seg, 'normal')

            init_weights(self.score_64, 'normal')
            init_weights(self.score_128, 'normal')
            init_weights(self.score_256, 'normal')
            init_weights(self.head, 'normal')

        else:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def build_upsample_content_layers(self, dims):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        self.up1_depth = Conc_Up_Residual(dims[4], dims[3], norm=norm)
        self.up2_depth = Conc_Up_Residual(dims[3], dims[2], norm=norm)
        self.up3_depth = Conc_Up_Residual(dims[2], dims[1], norm=norm)
        self.up4_depth = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)

        self.up1_seg = Conc_Up_Residual(dims[4], dims[3], norm=norm)
        self.up2_seg = Conc_Up_Residual(dims[3], dims[2], norm=norm)
        self.up3_seg = Conc_Up_Residual(dims[2], dims[1], norm=norm)
        self.up4_seg = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)

        self.up_depth = nn.Sequential(
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        self.up_seg = nn.Sequential(
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, source=None, target_1=None, target_2=None, label=None, out_keys=None, phase='train', content_layers=None,
                return_losses=True):
        result = {}
        layer_0 = self.relu(self.bn1(self.conv1(source)))
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        if self.trans:
            # content model branch
            up1_depth = self.up1_depth(layer_4, layer_3)
            up2_depth = self.up2_depth(up1_depth, layer_2)
            up3_depth = self.up3_depth(up2_depth, layer_1)
            up4_depth = self.up4_depth(up3_depth)
            result['gen_depth'] = self.up_depth(up4_depth)

            up1_seg = self.up1_seg(layer_4, layer_3)
            up2_seg = self.up2_seg(up1_seg, layer_2)
            up3_seg = self.up3_seg(up2_seg, layer_1)
            up4_seg = self.up4_seg(up3_seg)
            result['gen_seg'] = self.up_seg(up4_seg)

        # segmentation branch
        score_512 = self.head(layer_4)

        score_256 = None
        score_128 = None
        score_64 = None
        if self.cfg.WHICH_SCORE == 'main':
            score_256 = self.score_256(layer_3)
            score_128 = self.score_128(layer_2)
            score_64 = self.score_64(layer_1)
        elif self.cfg.WHICH_SCORE == 'up':
            # score_256_depth = self.score_256_depth(up1_depth)
            # score_128_depth = self.score_128_depth(up2_depth)
            # score_64_depth = self.score_64_depth(up3_depth)
            #
            # score_256_seg = self.score_256_seg(up1_seg)
            # score_128_seg = self.score_128_seg(up2_seg)
            # score_64_seg = self.score_64_seg(up3_seg)

            score_256 = self.score_256(torch.cat((up1_depth, up1_seg), 1))
            score_128 = self.score_128(torch.cat((up2_depth, up2_seg), 1))
            score_64 = self.score_64(torch.cat((up3_depth, up3_seg), 1))

        score = F.interpolate(score_512, score_256.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_256
        score = F.interpolate(score, score_128.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_128
        score = F.interpolate(score, score_64.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_64

        result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

        if 'SEMANTIC' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_content_depth'] = self.content_model(result['gen_depth'], target_1, layers=content_layers)
            result['loss_content_seg'] = self.content_model(result['gen_seg'], target_2, layers=content_layers)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_pix2pix_depth'] = self.pix2pix_criterion(result['gen_depth'], target_1)
            result['loss_pix2pix_seg'] = self.pix2pix_criterion(result['gen_seg'], target_2)

        if 'CLS' in self.cfg.LOSS_TYPES:
            result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


class FCN_Conc_Multiscale(nn.Module):

    def __init__(self, cfg, device=None):
        super(FCN_Conc_Multiscale, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        self.head = _FCNHead(512, num_classes, nn.BatchNorm2d)

        self.score_256 = nn.Conv2d(256, num_classes, 1)
        self.score_128 = nn.Conv2d(128, num_classes, 1)
        self.score_64 = nn.Conv2d(64, num_classes, 1)

        if self.trans:
            self.build_upsample_content_layers(dims, num_classes)

        if pretrained:
            if self.trans:
                # init_weights(self.up_image_14, 'normal')
                init_weights(self.up_image_28, 'normal')
                init_weights(self.up_image_56, 'normal')
                init_weights(self.up_image_112, 'normal')
                init_weights(self.up_image_224, 'normal')
                init_weights(self.up1, 'normal')
                init_weights(self.up2, 'normal')
                init_weights(self.up3, 'normal')
                init_weights(self.up4, 'normal')

            init_weights(self.head, 'normal')
            init_weights(self.score_64, 'normal')
            init_weights(self.score_128, 'normal')
            init_weights(self.score_256, 'normal')

        elif not pretrained:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def build_upsample_content_layers(self, dims, num_classes):

        # norm = nn.InstanceNorm2d
        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        self.up1 = Conc_Up_Residual(dims[4], dims[3], norm=norm)
        self.up2 = Conc_Up_Residual(dims[3], dims[2], norm=norm)
        self.up3 = Conc_Up_Residual(dims[2], dims[1], norm=norm)
        self.up4 = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)

        self.up_image_28 = nn.Sequential(
            nn.Conv2d(dims[3], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )
        self.up_image_56 = nn.Sequential(
            nn.Conv2d(dims[2], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )
        self.up_image_112 = nn.Sequential(
            nn.Conv2d(dims[1], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )
        self.up_image_224 = nn.Sequential(
            nn.Conv2d(dims[1], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )


    def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
                return_losses=True):
        result = {}
        layer_0 = self.relu(self.bn1(self.conv1(source)))
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        # content model branch
        if self.trans:

            scale_times = self.cfg.MULTI_SCALE_NUM
            ms_compare = []

            up1 = self.up1(layer_4, layer_3)
            up2 = self.up2(up1, layer_2)
            up3 = self.up3(up2, layer_1)
            up4 = self.up4(up3)

            compare_28 = self.up_image_28(up1)
            compare_56 = self.up_image_56(up2)
            compare_112 = self.up_image_112(up3)
            compare_224 = self.up_image_224(up4)

            ms_compare.append(compare_224)
            ms_compare.append(compare_112)
            ms_compare.append(compare_56)
            ms_compare.append(compare_28)
            # ms_compare.append(compare_14)
            # ms_compare.append(compare_7)

            result['gen_img'] = ms_compare[:scale_times]

        # segmentation branch
        score_512 = self.head(layer_4)

        score_256 = None
        score_128 = None
        score_64 = None
        if self.cfg.WHICH_SCORE == 'main':
            score_256 = self.score_256(layer_3)
            score_128 = self.score_128(layer_2)
            score_64 = self.score_64(layer_1)
        elif self.cfg.WHICH_SCORE == 'up':
            score_256 = self.score_256(up1)
            score_128 = self.score_128(up2)
            score_64 = self.score_64(up3)
        elif self.cfg.WHICH_SCORE == 'both':
            score_256 = self.score_256(up1 + layer_3)
            score_128 = self.score_128(up2 + layer_2)
            score_64 = self.score_64(up3 + layer_1)

        score = F.interpolate(score_512, score_256.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_256
        score = F.interpolate(score, score_128.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_128
        score = F.interpolate(score, score_64.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_64

        result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

        if self.trans and phase == 'train':
            scale_times = self.cfg.MULTI_SCALE_NUM
            trans_loss_list = []
            loss_key = None
            for i, (gen, _target) in enumerate(zip(result['gen_img'], target)):
                assert (gen.size()[-1] == _target.size()[-1])
                # content_layers = [str(layer) for layer in range(5 - i)]
                if 'SEMANTIC' in self.cfg.LOSS_TYPES:
                    loss_key = 'loss_content'
                    trans_loss_list.append(self.content_model(gen, _target, layers=content_layers))
                    
                elif 'PIX2PIX' in self.cfg.LOSS_TYPES:
                    loss_key = 'loss_pix2pix'
                    trans_loss_list.append(self.pix2pix_criterion(gen, _target))

            loss_coef = [1] * scale_times
            ms_losses = [loss_coef[i] * loss for i, loss in enumerate(trans_loss_list)]
            result[loss_key] = sum(ms_losses)
            
        if 'CLS' in self.cfg.LOSS_TYPES:
            result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)

#######################################################################
class UNet(nn.Module):
    def __init__(self, cfg, device=None):
        super(UNet, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

        self.score = nn.Conv2d(dims[1], num_classes, 1)

        # norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
        self.up1 = Conc_Up_Residual(dims[4], dims[3], norm=nn.BatchNorm2d)
        self.up2 = Conc_Up_Residual(dims[3], dims[2], norm=nn.BatchNorm2d)
        self.up3 = Conc_Up_Residual(dims[2], dims[1], norm=nn.BatchNorm2d)
        self.up4 = Conc_Up_Residual(dims[1], dims[1], norm=nn.BatchNorm2d, conc_feat=False)

        if pretrained:
            init_weights(self.up1, 'normal')
            init_weights(self.up2, 'normal')
            init_weights(self.up3, 'normal')
            init_weights(self.up4, 'normal')
            init_weights(self.score, 'normal')

        else:

            init_weights(self, 'normal')

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
                return_losses=True):
        result = {}

        layer_1 = self.layer1(self.relu(self.bn1(self.conv1(source))))
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        up1 = self.up1(layer_4, layer_3)
        up2 = self.up2(up1, layer_2)
        up3 = self.up3(up2, layer_1)
        up4 = self.up4(up3)

        result['cls'] = self.score(up4)
        result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result

class UNet_Long(nn.Module):
    def __init__(self, cfg, device=None):
        super(UNet_Long, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

        self.score = nn.Conv2d(dims[1], num_classes, 1)

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        self.up1 = Conc_Up_Residual(dims[4], dims[3], norm=norm)
        self.up2 = Conc_Up_Residual(dims[3], dims[2], norm=norm)
        self.up3 = Conc_Up_Residual(dims[2], dims[1], norm=norm)
        self.up4 = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)

        if self.trans:

            self.up_image_content = nn.Sequential(
                conv_norm_relu(64, 64, norm=norm),
                nn.Conv2d(64, 3, 7, 1, 3, bias=False),
                nn.Tanh()
            )

        if pretrained:
            init_weights(self.up1, 'normal')
            init_weights(self.up2, 'normal')
            init_weights(self.up3, 'normal')
            init_weights(self.up4, 'normal')
            init_weights(self.score, 'normal')

            if self.trans:
                init_weights(self.up_image_content, 'normal')
        else:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
                return_losses=True):
        result = {}

        layer_1 = self.layer1(self.relu(self.bn1(self.conv1(source))))
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        up1 = self.up1(layer_4, layer_3)
        up2 = self.up2(up1, layer_2)
        up3 = self.up3(up2, layer_1)
        up4 = self.up4(up3)

        if self.trans:
            result['gen_img'] = self.up_image_content(up4)

        result['cls'] = self.score(up4)

        if 'SEMANTIC' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)

        if 'CLS' in self.cfg.LOSS_TYPES:
            result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


class UNet_Share_256(nn.Module):
    def __init__(self, cfg, device=None):
        super(UNet_Share_256, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

        self.score = nn.Conv2d(dims[1], num_classes, 1)

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        self.up1 = Conc_Up_Residual(dims[4], dims[3])
        self.up2 = Conc_Up_Residual(dims[3], dims[2])
        self.up3 = Conc_Up_Residual(dims[2], dims[1])
        self.up4 = Conc_Up_Residual(dims[1], dims[1], conc_feat=False)

        if self.trans:
            self.up2_content = Conc_Up_Residual(dims[3], dims[2])
            self.up3_content = Conc_Up_Residual(dims[2], dims[1], norm=norm)
            self.up4_content = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)
            self.up_image = nn.Sequential(
                nn.Conv2d(64, 3, 7, 1, 3, bias=False),
                nn.Tanh()
            )

        if pretrained:
            init_weights(self.up1, 'normal')
            init_weights(self.up2, 'normal')
            init_weights(self.up3, 'normal')
            init_weights(self.up4, 'normal')
            init_weights(self.score, 'normal')

            if self.trans:
                init_weights(self.up2_content, 'normal')
                init_weights(self.up3_content, 'normal')
                init_weights(self.up4_content, 'normal')
                init_weights(self.up_image, 'normal')
        else:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
                return_losses=True):
        result = {}

        layer_1 = self.layer1(self.relu(self.bn1(self.conv1(source))))
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        up1 = self.up1(layer_4, layer_3)
        up2 = self.up2(up1, layer_2)
        up3 = self.up3(up2, layer_1)
        up4 = self.up4(up3)

        if self.trans:
            up2_content = self.up2_content(up1, layer_2)
            up3_content = self.up3_content(up2_content, layer_1)
            up4_content = self.up4_content(up3_content)
            result['gen_img'] = self.up_image(up4_content)

        result['cls'] = self.score(up4)

        if 'SEMANTIC' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)

        if 'CLS' in self.cfg.LOSS_TYPES:
            result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


class UNet_Share_128(nn.Module):
    def __init__(self, cfg, device=None):
        super(UNet_Share_128, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

        self.score = nn.Conv2d(dims[1], num_classes, 1)

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        self.up1 = Conc_Up_Residual(dims[4], dims[3])
        self.up2 = Conc_Up_Residual(dims[3], dims[2])
        self.up3 = Conc_Up_Residual(dims[2], dims[1])
        self.up4 = Conc_Up_Residual(dims[1], dims[1], conc_feat=False)

        if self.trans:

            self.up3_content = Conc_Up_Residual(dims[2], dims[1], norm=norm)
            self.up4_content = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)
            self.up_image = nn.Sequential(
                nn.Conv2d(64, 3, 7, 1, 3, bias=False),
                nn.Tanh()
            )

        if pretrained:
            init_weights(self.up1, 'normal')
            init_weights(self.up2, 'normal')
            init_weights(self.up3, 'normal')
            init_weights(self.up4, 'normal')
            init_weights(self.score, 'normal')

            if self.trans:
                init_weights(self.up3_content, 'normal')
                init_weights(self.up4_content, 'normal')
                init_weights(self.up_image, 'normal')
        else:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
                return_losses=True):
        result = {}

        layer_1 = self.layer1(self.relu(self.bn1(self.conv1(source))))
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        up1 = self.up1(layer_4, layer_3)
        up2 = self.up2(up1, layer_2)
        up3 = self.up3(up2, layer_1)
        up4 = self.up4(up3)

        if self.trans:
            # content
            up3_content = self.up3_content(up2, layer_1)
            up4_content = self.up4_content(up3_content)
            result['gen_img'] = self.up_image(up4_content)

        result['cls'] = self.score(up4)

        if 'SEMANTIC' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)

        if 'CLS' in self.cfg.LOSS_TYPES:
            result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


class UNet_Share_64(nn.Module):
    def __init__(self, cfg, device=None):
        super(UNet_Share_64, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

        self.score = nn.Conv2d(dims[1], num_classes, 1)

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        self.up1 = Conc_Up_Residual(dims[4], dims[3])
        self.up2 = Conc_Up_Residual(dims[3], dims[2])
        self.up3 = Conc_Up_Residual(dims[2], dims[1])
        self.up4 = Conc_Up_Residual(dims[1], dims[1], conc_feat=False)

        if self.trans:
            self.up4_content = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)
            self.up_image = nn.Sequential(
                nn.Conv2d(64, 3, 7, 1, 3, bias=False),
                nn.Tanh()
            )

        if pretrained:
            init_weights(self.up1, 'normal')
            init_weights(self.up2, 'normal')
            init_weights(self.up3, 'normal')
            init_weights(self.up4, 'normal')
            init_weights(self.score, 'normal')

            if self.trans:
                init_weights(self.up4_content, 'normal')
                init_weights(self.up_image, 'normal')
        else:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
                return_losses=True):
        result = {}
        layer_1 = self.layer1(self.relu(self.bn1(self.conv1(source))))
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        up1 = self.up1(layer_4, layer_3)
        up2 = self.up2(up1, layer_2)
        up3 = self.up3(up2, layer_1)
        up4 = self.up4(up3)

        if self.trans:
            # content
            up4_content = self.up4_content(up3)
            result['gen_img'] = self.up_image(up4_content)

        result['cls'] = self.score(up4)

        if 'SEMANTIC' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)

        if 'CLS' in self.cfg.LOSS_TYPES:
            result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)
class PSP_Conc_beachmark(nn.Module):
    def __init__(self,cfg,device=None, SyncBatchNorm=None,layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=19, zoom_factor=8, use_ppm=True, pretrained=True):
        super(PSP_Conc_beachmark, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.device = device
        self.use_ppm = use_ppm
        self.cfg=cfg
        self.using_semantic_branch = not cfg.NO_TRANS
        models.BatchNorm = SyncBatchNorm
        BatchNorm=SyncBatchNorm
        self.norm=SyncBatchNorm
        dims = [32, 64, 128, 256, 512, 1024, 2048]


        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
            print("load resnet50")
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
            print("load resnet101")
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu,resnet.maxpool)
        # self.maxpool= resnet.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        if self.using_semantic_branch:
            self.build_upsample_content_layers(dims)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        # if self.training:
        self.aux = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(256, classes, kernel_size=1)
        )

        if self.using_semantic_branch:
            init_weights(self.up1, 'normal')
            init_weights(self.up2, 'normal')
            init_weights(self.up3, 'normal')
            init_weights(self.up4, 'normal')
            init_weights(self.up5, 'normal')
            init_weights(self.up_seg, 'normal')

        init_weights(self.aux, 'normal')
        init_weights(self.cls, 'normal')
    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)
    def set_content_model(self, content_model):
        self.content_model = content_model
    def build_upsample_content_layers(self, dims):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
        # norm=self.norm
        self.up1 = Conc_Up_Residual_bottleneck(dims[6], dims[5], norm=norm)
        self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
        self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
        self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[3], norm=norm, conc_feat=False)
        self.up5 = Conc_Up_Residual_bottleneck(dims[3], dims[3], norm=norm, conc_feat=False)

        self.up_seg = nn.Sequential(
            nn.Conv2d(256, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )
    def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
                return_losses=True):
        result={}
        x=source
        y=label
        layer_0=self.layer0(x)
        # if not self.using_semantic_branch:
        # layer_0 = self.maxpool(layer_0)
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)


        if self.use_ppm:
            x = self.ppm(layer_4)
        x = self.cls(x)
        result['cls'] = F.interpolate(x, source.size()[2:], mode='bilinear', align_corners=True)
        if self.cfg.SLIDE_WINDOWS==True and phase=='test':
            return result['cls']
        if phase=='train':
            aux = self.aux(layer_3)
            aux = F.interpolate(aux,source.size()[2:], mode='bilinear', align_corners=True)
            main_loss = self.cls_criterion(result['cls'], y)
            aux_loss = self.cls_criterion(aux, y)
            result['loss_cls'] = main_loss+0.4*aux_loss
        if phase=='test':
            main_loss = self.cls_criterion(result['cls'], y)
            result['loss_cls'] = main_loss
        if self.using_semantic_branch and phase == 'train':
            up1_seg = self.up1(layer_4, layer_3)
            up2_seg = self.up2(up1_seg, layer_2)
            up3_seg = self.up3(up2_seg, layer_1)
            up4_seg = self.up4(up3_seg)
            up5_seg = self.up5(up4_seg)
            result['gen_img'] = self.up_seg(up5_seg)
            result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

        return result