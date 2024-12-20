import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C2f, C2fCIB, SCDown, Bottleneck
from ultralytics.nn.modules.conv import Concat, Conv

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS

from mmcv.cnn import ConvModule
from ..utils import resize

from PCFN import PCFN

from AFMA import EncoderChannels, AFMA


@MODELS.register_module()
class yolo_head_dynamic(BaseDecodeHead):
    def __init__(self,
                 in_channels,
                 channels,
                 num_classes=2,
                 interpolate_mode='bilinear',
                 **kwargs):
        """
            x1, x2, x3, x4分别为四个Tensor的通道数量
        """
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            input_transform='multiple_select',
            **kwargs
        )

        self.num_classes = num_classes
        self.channel_num_1 = in_channels[0]  # (_, 80, 56, 56)
        self.channel_num_2 = in_channels[1]  # (_, 160, 28, 28)
        self.channel_num_3 = in_channels[2]  # (_, 320, 14, 14)
        self.channel_num_4 = in_channels[3]  # (_, 640, 7, 7)

        # 上采样，第一部分的模块定义
        self.C2f_3 = C2f(
            self.channel_num_4 + self.channel_num_3,
            self.channel_num_4
        )

        self.C2f_4 = C2f(
            self.channel_num_2 + self.channel_num_4,
            self.channel_num_3
        )

        self.C2f_5 = C2f(
            self.channel_num_1 + self.channel_num_3,
            self.channel_num_2
        )
        self.PCFN1 = PCFN(self.channel_num_2)
        self.PCFN2 = PCFN(self.channel_num_3)
        self.PCFN3 = PCFN(self.channel_num_4)

        # 下采样，第二部分的模块定义
        self.C2fCIB_2 = C2fCIB(
            c1=self.channel_num_3 + self.channel_num_4,
            c2=self.channel_num_4
        )

        self.C2fCIB_3 = C2fCIB(
            c1=self.channel_num_4 * 2,
            c2=self.channel_num_4
        )

        self.SCDown = SCDown(
            c1=self.channel_num_4,
            c2=self.channel_num_4,
            k=3, s=2
        )

        self.Conv = Conv(
            self.channel_num_3,
            self.channel_num_3,
            k=3, s=2
        )

        # 通用模块
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # 2x2 --> 4x4

        self.Concat = Concat(dimension=1)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        yolo_channel = [
            self.channel_num_2, self.channel_num_3, self.channel_num_4, self.channel_num_4
        ]

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=yolo_channel[i],  # 输入channel变为yolo处理后的channel
                    out_channels=self.channels,  # 输出channel不变
                    kernel_size=1,
                    stride=1
                )
            )

        self.fusion_conv = ConvModule(
            in_channels=self.channels * 4,
            out_channels=self.channels,
            kernel_size=1,
        )

        self.weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))

        self.attention = EncoderChannels(self.channel_num_1)
        self.AFMA = AFMA(self.num_classes)

    def forward(self, features):
        """
            x1 = torch.randn(_, 80, 56, 56)
            x2 = torch.randn(_, 160, 28, 28)
            x3 = torch.randn(_, 320, 14, 14)
            x4 = torch.randn(_, 640, 7, 7)
        """
        x1, x2, x3, x4 = features

        # 上采样，第一部分
        p0 = self.upsample(x4)  # (_, 640, 14, 14)
        p0_Concat = self.Concat([p0, x3])  # (_, 960, 14, 14)
        p0_Concat_C2f = self.C2f_3(p0_Concat)  # (_, 640, 14, 14)
        p0_Concat_C2f = self.PCFN3(p0_Concat_C2f)

        p1 = self.upsample(p0_Concat_C2f)  # (_, 640, 28, 28)
        p1_Concat = self.Concat([p1, x2])  # (_, 800, 28, 28)
        p1_Concat_C2f = self.C2f_4(p1_Concat)  # (_, 320, 28, 28)  # head2
        p1_Concat_C2f = self.PCFN2(p1_Concat_C2f)

        # 对yolo的拓展,额外部分
        p_add = self.upsample(p1_Concat_C2f)  # (_, 320, 56, 56)
        p_add_Concat = self.Concat([p_add, x1])  # (_, 400, 56, 56)
        p_add_Concat_C2f = self.C2f_5(p_add_Concat)  # (_, 160. 56, 56)  # head1
        p_add_Concat_C2f = self.PCFN1(p_add_Concat_C2f)

        # 下采样，第二部分
        p2 = self.Conv(p1_Concat_C2f)  # (_, 320, 14, 14)
        p2_Concat = self.Concat([p2, p0_Concat_C2f])  # (_, 960, 14, 14)
        p2_Concat_C2fCIB = (
            self.C2fCIB_2(p2_Concat))  # (_, 640, 14, 14)  # head3

        p3 = self.SCDown(p2_Concat_C2fCIB)  # (_, 640, 7, 7)
        p3_Concat = (
            self.Concat([p3, x4]))  # (_, 1280, 7, 7)
        p3_Concat_C2fCIB = (
            self.C2fCIB_3(p3_Concat))  # (_, 640, 7, 7)    # head4

        inputs = [
            p_add_Concat_C2f,  # (_, 160. 56, 56)  # head1
            p1_Concat_C2f,  # (_, 320, 28, 28)  # head2
            p2_Concat_C2fCIB,  # (_, 640, 14, 14)  # head3
            p3_Concat_C2fCIB  # (_, 640, 7, 7)    # head4
        ]

        outs = []

        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            out_resized = resize(
                input=conv(x),
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners
            )

            # 对每个输出乘以对应的权重，根据特征图的宽高大小进行调整
            dynamic_weight = (x.shape[2] * x.shape[3]) ** -0.5
            outs.append(out_resized * self.weights[idx] * (1 + dynamic_weight))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.cls_seg(out)

        attention = self.attention(x1)
        seg_out, _ = self.AFMA(out, attention)

        return seg_out


@MODELS.register_module()
class yolo_head(BaseDecodeHead):
    def __init__(self,
                 in_channels,
                 channels,
                 num_classes=2,
                 interpolate_mode='bilinear',
                 **kwargs):
        """
            x1, x2, x3, x4分别为四个Tensor的通道数量
        """
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            input_transform='multiple_select',
            **kwargs
        )

        self.num_classes = num_classes
        self.channel_num_1 = in_channels[0]  # (_, 80, 56, 56)
        self.channel_num_2 = in_channels[1]  # (_, 160, 28, 28)
        self.channel_num_3 = in_channels[2]  # (_, 320, 14, 14)
        self.channel_num_4 = in_channels[3]  # (_, 640, 7, 7)

        # 上采样，第一部分的模块定义
        self.C2f_3 = C2f(
            c1=self.channel_num_4 + self.channel_num_3,
            c2=self.channel_num_4
        )
        self.C2f_4 = C2f(
            c1=self.channel_num_2 + self.channel_num_4,
            c2=self.channel_num_3
        )
        self.C2f_5 = C2f(
            c1=self.channel_num_1 + self.channel_num_3,
            c2=self.channel_num_2
        )

        # 下采样，第二部分的模块定义
        self.C2fCIB_2 = C2fCIB(
            c1=self.channel_num_3 + self.channel_num_4,
            c2=self.channel_num_4
        )
        self.C2fCIB_3 = C2fCIB(
            c1=self.channel_num_4 * 2,
            c2=self.channel_num_4
        )
        self.SCDown = SCDown(
            c1=self.channel_num_4,
            c2=self.channel_num_4,
            k=3, s=2
        )
        self.Conv = Conv(
            self.channel_num_3,
            self.channel_num_3,
            k=3, s=2
        )

        # 通用模块
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # 2x2 --> 4x4
        self.Concat = Concat(dimension=1)

        self.interpolate_mode = interpolate_mode
        num_inputs = 3  # len(self.in_channels)

        yolo_channel = [
            # self.channel_num_2,
            self.channel_num_3, self.channel_num_4, self.channel_num_4
        ]

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=yolo_channel[i],  # 输入channel变为yolo处理后的channel
                    out_channels=self.channels,  # 输出channel不变
                    kernel_size=1,
                    stride=1
                )
            )

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
        )

    def forward(self, features):
        """
            x1 = torch.randn(_, 80, 56, 56)
            x2 = torch.randn(_, 160, 28, 28)
            x3 = torch.randn(_, 320, 14, 14)
            x4 = torch.randn(_, 640, 7, 7)
        """
        x1, x2, x3, x4 = features

        # 上采样，第一部分
        p0 = self.upsample(x4)  # (_, 640, 14, 14)
        p0_Concat = self.Concat([p0, x3])  # (_, 960, 14, 14)
        p0_Concat_C2f = self.C2f_3(p0_Concat)  # (_, 640, 14, 14)

        p1 = self.upsample(p0_Concat_C2f)  # (_, 640, 28, 28)
        p1_Concat = self.Concat([p1, x2])  # (_, 800, 28, 28)
        p1_Concat_C2f = self.C2f_4(p1_Concat)  # (_, 320, 28, 28)  # head2

        # 对yolo的拓展,额外部分
        p_add = self.upsample(p1_Concat_C2f)  # (_, 320, 56, 56)
        p_add_Concat = self.Concat([p_add, x1])  # (_, 400, 56, 56)
        p_add_Concat_C2f = self.C2f_5(p_add_Concat)  # (_, 160. 56, 56)  # head1

        # 下采样，第二部分
        p2 = self.Conv(p1_Concat_C2f)  # (_, 320, 14, 14)
        p2_Concat = self.Concat([p2, p0_Concat_C2f])  # (_, 960, 14, 14)
        p2_Concat_C2fCIB = (
            self.C2fCIB_2(p2_Concat))  # (_, 640, 14, 14)  # head3

        p3 = self.SCDown(p2_Concat_C2fCIB)  # (_, 640, 7, 7)
        p3_Concat = (
            self.Concat([p3, x4]))  # (_, 1280, 7, 7)
        p3_Concat_C2fCIB = (
            self.C2fCIB_3(p3_Concat))  # (_, 640, 7, 7)    # head4

        inputs = [
            # p_add_Concat_C2f,  # (_, 160. 56, 56)  # head1
            p1_Concat_C2f,  # (_, 320, 28, 28)  # head2
            p2_Concat_C2fCIB,  # (_, 640, 14, 14)  # head3
            p3_Concat_C2fCIB  # (_, 640, 7, 7)    # head4
        ]

        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.cls_seg(out)

        return out
