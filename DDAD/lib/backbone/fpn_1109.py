import math

import torch
from torch import nn
import torch.nn.functional as F

class FPN(nn.Module):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """
    def __init__(self, bottom_up, layers_begin, layers_end, return_base_f= False):
        super(FPN, self).__init__()
        assert layers_begin > 1 and layers_begin < 6
        assert layers_end > 4 and layers_begin < 8
        in_channels = [256, 512, 1024, 2048]
        fpn_dim = 256
        in_channels = in_channels[layers_begin-2:]

        
        self.return_base_f = return_base_f

        if self.return_base_f:
            density_lateral_convs = nn.ModuleList()
            density_output_convs = nn.ModuleList()
            for idx, in_channel in enumerate(in_channels):
                denisty_lateral_conv = nn.Conv2d(in_channel, fpn_dim, kernel_size=1)
                denisty_output_conv = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1)
                nn.init.kaiming_normal_(denisty_lateral_conv.weight, mode='fan_out')
                nn.init.constant_(denisty_lateral_conv.bias, 0)
                nn.init.kaiming_normal_(denisty_output_conv.weight, mode='fan_out')
                nn.init.constant_(denisty_output_conv.bias, 0)
                density_lateral_convs.append(denisty_lateral_conv)
                density_output_convs.append(denisty_output_conv)
            self.density_lateral_convs = density_lateral_convs[::-1]
            self.density_output_convs = density_output_convs[::-1]


        lateral_convs = nn.ModuleList()
        output_convs = nn.ModuleList()
        for idx, in_channels in enumerate(in_channels):
            lateral_conv = nn.Conv2d(in_channels, fpn_dim, kernel_size=1)
            output_conv = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1)
            nn.init.kaiming_normal_(lateral_conv.weight, mode='fan_out')
            nn.init.constant_(lateral_conv.bias, 0)
            nn.init.kaiming_normal_(output_conv.weight, mode='fan_out')
            nn.init.constant_(output_conv.bias, 0)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.bottom_up = bottom_up
        self.output_b = layers_begin
        self.output_e = layers_end
        if self.output_e == 7:
            self.p6 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=2, padding=1)
            self.p7 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=2, padding=1)
            for l in [self.p6, self.p7]:
                nn.init.kaiming_uniform_(l.weight, a=1)  # pyre-ignore
                nn.init.constant_(l.bias, 0)
        BN_MOMENTUM = 0.9
        self.denisty_2_fpn_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        ) 
        self.denisty_2_fpn_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.denisty_2_fpn_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.denisty_2_fpn_4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        spesial_conv  = True
        if spesial_conv:
            self.fpn_concat_back_1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3,1), stride=1, padding=(1,0)),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(1,3), stride=1, padding=(0,1)),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3,1), stride=1, padding=(1,0)),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),

            ) 
            self.fpn_concat_back_2 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3,1), stride=1, padding=(1,0)),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(1,3), stride=1, padding=(0,1)),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3,1), stride=1, padding=(1,0)),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),

            ) 
            self.fpn_concat_back_3 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3,1), stride=1, padding=(1,0)),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(1,3), stride=1, padding=(0,1)),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3,1), stride=1, padding=(1,0)),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),

            ) 
            self.fpn_concat_back_4 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3,1), stride=1, padding=(1,0)),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(1,3), stride=1, padding=(0,1)),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3,1), stride=1, padding=(1,0)),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            ) 
        else:
            self.fpn_concat_back_1 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),

            ) 
            self.fpn_concat_back_2 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),

            ) 
            self.fpn_concat_back_3 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),

            ) 
            self.fpn_concat_back_4 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),

            ) 


    def forward(self, x):

        bottom_up_features = self.bottom_up(x)
        last_f = bottom_up_features[-1]
        bottom_up_features = bottom_up_features[self.output_b - 2:]
        bottom_up_features = bottom_up_features[::-1]
        denisty_bottom_up_features = bottom_up_features

        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[0])
        results.append(self.output_convs[0](prev_features))
        for l_id, (features, lateral_conv, output_conv) in enumerate(zip(
            bottom_up_features[1:], self.lateral_convs[1:], self.output_convs[1:])):

            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="bilinear", align_corners=False)
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            results.append(output_conv(prev_features))

        if(self.output_e == 6):
            p6 = F.max_pool2d(results[0], kernel_size=1, stride=2, padding=0)
            results.insert(0, p6)
        elif(self.output_e == 7):
            p6 = self.p6(results[0])
            results.insert(0, p6)
            p7 = self.p7(F.relu(results[0]))
            results.insert(0, p7)

        if self.return_base_f :
            density_results = []
            prev_density_features = self.density_lateral_convs[0](denisty_bottom_up_features[0])
            density_results.append(self.density_output_convs[0](prev_density_features))

            for l_id, (features, lateral_conv, output_conv) in enumerate(zip(
                denisty_bottom_up_features[1:], self.density_lateral_convs[1:], self.density_output_convs[1:])):

                top_down_features = F.interpolate(prev_density_features, scale_factor=2, mode="bilinear", align_corners=False)
                lateral_features = lateral_conv(features)
                prev_density_features = lateral_features + top_down_features
                density_results.append(output_conv(prev_density_features))
                
        if self.return_base_f :
            results[-1] = F.relu(self.fpn_concat_back_4(torch.cat([self.denisty_2_fpn_1(density_results[-1]), results[-1]], axis=1)))
            results[-2] = F.relu(self.fpn_concat_back_3(torch.cat([self.denisty_2_fpn_2(density_results[-2].detach()), results[-2]], axis=1)))
            results[-3] = F.relu(self.fpn_concat_back_2(torch.cat([self.denisty_2_fpn_3(density_results[-3].detach()), results[-3]], axis=1)))
            results[-4] = F.relu(self.fpn_concat_back_1(torch.cat([self.denisty_2_fpn_4(density_results[-4].detach()), results[-4]], axis=1)))

            return density_results[-1], results
                
        return results
