import pdb
from contextlib import nullcontext

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import numpy as np
from models.clf import Classifier_Module
from utils_ import interpn

affine_par = True


def drop_path(x, drop_prob=0, training=False, scale_by_keep=True):
    if drop_prob == 0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None, droprate=0, droppath=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride , bias=False)
        self.bn1 = BatchNorm(planes, affine=affine_par)

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding, bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes, affine=affine_par)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.droprate = droprate
        if droprate > 0:
            self.dropout = nn.Dropout2d(droprate)
        if droppath > 0:
            self.droppath = DropPath(droppath)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if hasattr(self, "droppath"):
            out = residual + self.droppath(x)
        else:
            out += residual
        out = self.relu(out)
        if self.droprate > 0:
            out = self.dropout(out)

        return out


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim=256, out_dim=256):
        super(MLP, self).__init__()

        self.linear1 = nn.Conv2d(in_dim, hid_dim, 1)
        self.bn1 = nn.GroupNorm(num_groups=32, num_channels=hid_dim, affine=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 = nn.Conv2d(hid_dim, out_dim, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)

        return x


class ResNet101(nn.Module):
    def __init__(self, block, layers, output_stride, BatchNorm, mtype='Base', opt=None):
        self.name_ = self.__class__.__name__
        self.opt = opt
        self.inplanes = 64
        self.drop_layers = opt.drop_layers
        super(ResNet101, self).__init__()

        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError
        droprates = [opt.droprate if i in self.drop_layers else 0 for i in range(5)]
        droppaths = [opt.droppath if i in self.drop_layers else 0 for i in range(5)]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        if droprates[0] > 0:
            self.dropout = nn.Dropout2d(droprates[0])

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm, droprate=droprates[1], droppath=droppaths[1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm, droprate=droprates[2], droppath=droppaths[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm, droprate=droprates[3], droppath=droppaths[3])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm, droprate=droprates[4], droppath=droppaths[4])

        self.layer5 = Classifier_Module(2048, [6, 12, 18, 24], [6, 12, 18, 24], opt.n_class, opt.clf_droprate)

        ### projector (only for BaseNet) ###
        if mtype == 'Base' and opt.intra_consist:
            self.projector = MLP(256, hid_dim=256, out_dim=256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        ### Cross-domain consistency ###
        if self.opt.cross_consist:
            out_channels = [64, 256, 512, 1024, 2048]
            for i in self.opt.style_transfer_layers:
                self.register_buffer(f"features_mean_{i}", torch.randn(self.opt.n_class, self.opt.stat_dict_size, out_channels[i]))
                self.register_buffer(f"features_std_{i}", torch.randn(self.opt.n_class, self.opt.stat_dict_size, out_channels[i]))
                self.register_buffer(f"memory_ptr_{i}", torch.zeros(self.opt.n_class, dtype=torch.long))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None, droprate=0, droppath=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion, affine=affine_par))

        layers = [block(self.inplanes, planes, stride,
                        dilation=dilation, 
                        downsample=downsample, 
                        BatchNorm=BatchNorm,
                        droprate=droprate)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                dilation=dilation, 
                                BatchNorm=BatchNorm,
                                droprate=droprate,
                                droppath=droppath))

        return nn.Sequential(*layers)

    def forward(self, x, label=None, ret=-1, state='none', forward_from=-1, features_mean=None, features_std=None):
        context = torch.cuda.amp.autocast if self.opt.amp else nullcontext
        with context():
            return self._forward(x, label, ret, state, forward_from, features_mean, features_std)

    def _forward(self, x, label=None, ret=-1, state='none', forward_from=-1, features_mean=None, features_std=None):
        feats = {}

        ### Layer 0 ###
        if forward_from < 0:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            feats['f0'] = x
            if 0 in self.drop_layers:
                x = self.dropout(x)

        ### Layer 1 ###
        if forward_from < 1:
            x = self.maybe_transfer(x, label, 0, state, feats, features_mean, features_std)
            x = self.layer1(x)
            feats['f1'] = x

        ### Layer 2 ###
        if forward_from < 2:
            x = self.maybe_transfer(x, label, 1, state, feats, features_mean, features_std)
            x = self.layer2(x)
            feats['f2'] = x
        
        ### Layer 3 ###
        if forward_from < 3:
            x = self.maybe_transfer(x, label, 2, state, feats, features_mean, features_std)
            x = self.layer3(x)
            feats['f3'] = x
            
        ### Layer 4 ###
        if forward_from < 4:
            x = self.maybe_transfer(x, label, 3, state, feats, features_mean, features_std)
            x = self.layer4(x)
            feats['f4'] = x
        
        ### Layer 5 ###
        if forward_from < 5:
            x = self.maybe_transfer(x, label, 4, state, feats, features_mean, features_std)
            out = self.layer5(x)

        ### Predictor layer ###
        if hasattr(self, 'projector'):
            out['proj'] = self.projector(out['feat'])

        out.update(feats)
        return out

    def style_save_per_image(self, x, label, layer):
        C = x.size(1)
        features_mean = []
        features_std = []
        classes = []

        # sample and restore pixels to compute statistics when source images forwarding
        x = x.detach()
        with torch.no_grad():

            if x.shape[-2:] != label.shape[-2:]:
                label = interpn(label.unsqueeze(1).float(), x.shape[-2:]).squeeze(1).long()

            x = x.permute(0, 2, 3, 1)
            x = x.reshape(x.size(0), -1, x.size(3))             # [B, HW, C]
            label = label.view(label.size(0), -1)               # [B, HW]

            for ii in range(x.size(0)):
                x_i = x[ii]                                     # [HW, C]
                label_i = label[ii]                             # [HW]

                for c in torch.unique(label_i):
                    if c == 250:
                        continue
                    tmp_idx = torch.where(label_i == c)[0]
                    if tmp_idx.size(0) < 2:
                        # print("The class area is less than 2. Ignore this class in current image.")
                        continue
                    std, mean = torch.std_mean(x_i[tmp_idx], dim=0)   # [C]
                    
                    # Save to memory
                    features_mean.append(mean)  # [C]
                    features_std.append(std)    # [C]
                    classes.append(c.view(-1))  # [1]
        
        if len(classes) == 0:
            return (
                torch.zeros((0, C), dtype=x.dtype, device=x.device), 
                torch.zeros((0, C), dtype=x.dtype, device=x.device), 
                torch.zeros((0,), dtype=torch.long, device=x.device)
            )
        return (
            torch.stack(features_mean, dim=0),  # [?, C]
            torch.stack(features_std, dim=0),   # [?, C]
            torch.cat(classes, dim=0)           # [?]
        )

    def style_load_per_image(self, x, label, layer, eps=1e-5, features_mean=None, features_std=None):
        B, C, H, W = x.size()

        if x.shape[-2:] != label.shape[-2:]:
            label = interpn(label.unsqueeze(1).float(), x.shape[-2:]).squeeze(1).long()

        std_tgt = features_std     # shuffle outside the DataParallel
        mean_tgt = features_mean     # shuffle outside the DataParallel
        assert std_tgt.shape == (self.opt.n_class, B, C), f"{std_tgt.shape} vs {(self.opt.n_class, B, C)}"

        std_tgt = torch.cat((std_tgt, torch.ones((1, B, C), dtype=std_tgt.dtype, device=std_tgt.device)), dim=0)       # [n_cls + 1, B, C]
        mean_tgt = torch.cat((mean_tgt, torch.zeros((1, B, C), dtype=mean_tgt.dtype, device=mean_tgt.device)), dim=0)  # [n_cls + 1, B, C]

        # Get class-wise source domain statistics from current mini-batch
        with torch.no_grad():
            x_new = x.permute(0, 2, 3, 1)
            x_new = x_new.view(x_new.size(0), -1, x_new.size(3))        # [B, HW, C]
            label_new = label.view(label.size(0), -1)                   # [B, HW]

            aa_map = torch.zeros_like(x, dtype=x.dtype, device=x.device)
            bb_map = torch.zeros_like(x, dtype=x.dtype, device=x.device)

            for ii in range(B):
                x_new_i = x_new[ii]                 # [HW, C]
                label_new_i = label_new[ii]         # [HW]
                label_i = label[ii]                 # [H, W]

                std_tgt_i = std_tgt[:, ii]         # [n_cls + 1, C]
                mean_tgt_i = mean_tgt[:, ii]       # [n_cls + 1, C]

                std_src_i = torch.ones_like(std_tgt_i, dtype=std_tgt.dtype, device=std_tgt.device)
                mean_src_i = torch.zeros_like(mean_tgt_i, dtype=mean_tgt.dtype, device=mean_tgt.device)

                for c in torch.unique(label_new_i):
                    if c == 250:
                        continue
                    tmp_idx = torch.where(label_new_i == c)[0]

                    if tmp_idx.size(0) < 2:
                        # print("The class area is less than 2. Ignore this class in current image.")
                        continue
                    cls_x = x_new_i[tmp_idx]
                    std, mean = torch.std_mean(cls_x, dim=0)
                    std_src_i[c] = std
                    mean_src_i[c] = mean

                ######################################################################
                #        (x - mean_src)
                #       ---------------- * std_tgt + mean_tgt
                #            std_src
                # 
                #            std_tgt                              std_tgt
                #  ==  x * ----------- + (mean_tgt - mean_src * -----------)
                #            std_src                              std_src
                # 
                #  ==  x * aa + bb
                # 
                ######################################################################

                aa = std_tgt_i / (std_src_i + eps)
                torch.clamp_(aa, -10000, 10000)
                bb = mean_tgt_i - mean_src_i * aa

                # construct std & mean maps
                label_i = label_i.clone()                                   # [H, W]
                label_i[label_i == 250] = self.opt.n_class                  # [H, W], values from [0, n_cls + 1)
                aa_map[ii] = aa[label_i].permute(2, 0, 1).contiguous()      # [C, H, W]
                bb_map[ii] = bb[label_i].permute(2, 0, 1).contiguous()      # [C, H, W]

            # transfer source images to target domain
            x = x * aa_map + bb_map

        return x
    
    def maybe_transfer(self, x, label, layer, state, feat_dict, features_mean=None, features_std=None):
        if self.opt.cross_consist and layer in self.opt.style_transfer_layers:
            if state == 'style_save':
                means, stds, classes = self.style_save_per_image(x, label, layer)
                feat_dict[f'means{layer}'] = means
                feat_dict[f'stds{layer}'] = stds
                feat_dict[f'classes{layer}'] = classes
                return x
            elif state == 'style_load':
                return self.style_load_per_image(x, label, layer, features_mean=features_mean[layer][0], features_std=features_std[layer][0])
        return x

    def _get_1x_lr_params(self):
        b = [
            self.conv1, self.bn1,
            self.layer1, self.layer2, self.layer3, self.layer4,
        ]

        for i in range(len(b)):
            # ---------------------------------------------------------------------------------
            # Next lines has a bug coming from the ProDA (https://github.com/microsoft/ProDA):
            #   
            #   Specifically, each parameter may be iterated multiple times. It results in 
            #   these parameters have k times learning rate, where k is the number of 'dot'(.) 
            #   in their names.
            #    
            #   For example, the parameter named 'layer1.0.downsample.0.weight' has 4 dots in 
            #   its name, so this generator will yield this parameter 4 times, which equals to 
            #   that this parameter has a learning rate of 4 * lr. 
            # 
            #   This bug sometimes is a `feature`. When I correct it, recovering the same 
            #   performance by only adjusting the learning rate is nontrival. So I just keep it 
            #   unchanged. 
            #   
            #   Correction:
            #
            #   for layer in b:
            #       for k in layer.parameters():
            #           if k.requires_grad:
            #               yield k
            # ---------------------------------------------------------------------------------
            for j in b[i].modules():
                for k in j.parameters():
                    if k.requires_grad:
                        yield k

    def _get_10x_lr_params(self):
        b = [
            self.layer5
        ]

        for j in range(len(b)):
            for k in b[j].parameters():
                yield k

    def _get_encoder_params(self):
        b = [
            self.conv1, self.bn1,
            self.layer1, self.layer2, self.layer3, self.layer4, self.layer5
        ]

        for i in range(len(b)):
            for j in b[i].modules():
                for k in j.parameters():
                    if k.requires_grad:
                        yield k

    def _get_projector_params(self):
        return self.projector.parameters()

    def get_param_list(self, stage, base_lr, proj_lr_const=True):
        if stage in ["warm_up", "source_only", "target_only"]:
            param_list = [
                {'params': self._get_1x_lr_params(), 'lr': base_lr},
                {'params': self._get_10x_lr_params(), 'lr': base_lr * 10}
            ]
        elif stage == "self_training":
            # Following SimSiam
            if proj_lr_const:
                param_list = [
                    {'params': self._get_encoder_params(), 'lr': base_lr},
                    {'params': self._get_projector_params(), 'lr': base_lr}
                ]
            else:
                param_list = self.parameters()
        else:
            raise ValueError(f"`stage` only accept warm_up|source_only|target_only|self_training, got {stage}")

        return param_list


def Deeplab(BatchNorm,
            output_stride,
            mtype='Base',
            opt=None):

    ### Construct model ###
    model = ResNet101(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, mtype=mtype, opt=opt)

    return model
