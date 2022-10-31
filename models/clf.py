import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, inplanes, r=16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.se = nn.Sequential(
                nn.Linear(inplanes, inplanes // r),
                nn.ReLU(inplace=True),
                nn.Linear(inplanes // r, inplanes),
                nn.Sigmoid()
        )

    def forward(self, x):
        xx = self.global_pool(x)
        xx = xx.view(xx.size(0), xx.size(1))
        se_weight = self.se(xx).unsqueeze(-1).unsqueeze(-1)
        return x.mul(se_weight)


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes, clf_droprate=0.1):
        super(Classifier_Module, self).__init__()

        self.conv2d_list = nn.ModuleList()
        self.conv2d_list.append(
                nn.Sequential(
                    nn.Conv2d(inplanes, 256, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                    nn.GroupNorm(num_groups=32, num_channels=256, affine=True),
                    nn.ReLU(inplace=True)
                ))

        # ASPP
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Sequential(
                    nn.Conv2d(inplanes, 256, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True),
                    nn.GroupNorm(num_groups=32, num_channels=256, affine = True),
                    nn.ReLU(inplace=True)
                ))
 
        self.bottleneck = nn.Sequential(
            SEBlock(256 * (len(dilation_series) + 1)),
            nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) ,
            nn.GroupNorm(num_groups=32, num_channels=256, affine = True)
        )

        self.head = nn.Sequential(
            nn.Dropout2d(clf_droprate),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=False)
        )
        
        ### initialization ###
        for m in self.conv2d_list:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.bottleneck:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.head:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out = torch.cat( (out, self.conv2d_list[i + 1](x)), 1)
        out = self.bottleneck(out)

        out_dict = {}

        out = self.head[0](out)
        out_dict['feat'] = out

        out = self.head[1](out)
        out_dict['out'] = out

        return out_dict
