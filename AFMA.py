import torch.nn as nn
import torch


class EncoderChannels(nn.Module):
    def __init__(self, in_channels, classes_num=2, patch_size=10):
        super().__init__()

        # 在源码中，in_channels被设为3，即原始图片的通道数
        # 在这里，将in_channels设为输入特征的大小
        self.in_channels = in_channels

        self.patch_size = patch_size

        # 用于处理输入的原始图片/特征
        self.conv_img = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), padding=3),
            nn.Conv2d(64, classes_num, kernel_size=(3, 3), padding=1)
        )

        # 特征图卷积层，用于将特征图映射到classes_num（与特征图的列数相同）
        # 具体可见论文Fig.3
        self.conv_feamap = nn.Sequential(
            nn.Conv2d(self.in_channels, classes_num, kernel_size=(1, 1), stride=1)
        )

        # 展开操作，用于将图像分割为Patch
        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size),
                                stride=(self.patch_size, self.patch_size))

        # 转换层，用于对展开后的 patch 进行线性变换
        self.resolution_trans = nn.Sequential(
            nn.Linear(self.patch_size * self.patch_size, 2 * self.patch_size * self.patch_size, bias=False),
            nn.Linear(2 * self.patch_size * self.patch_size, self.patch_size * self.patch_size, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        attentions = []  # 存储每一层的注意力图

        ini_img = self.conv_img(x)  # 图像初步卷积处理      # 1, 12, 56, 56

        # 计算注意力
        # 将特征图通过卷积层进行处理，并缩放。
        # 源码中的缩放因子为 2^(attention_on_depth) 的平方，此处设置为了1
        # 目的是对不同层次的特征图进行归一化处理。
        feamap = self.conv_feamap(x) / (2 ** 1 * 2 ** 1)    # 1, 12, 56, 56

        # 遍历特征图的每个通道，逐通道进行注意力计算
        for i in range(feamap.size()[1]):
            # ————————————————————对应论文Fig.3中的1、3步骤————————————————————
            # 将初始卷积图（ini_img）的第 i 个通道展开成小块（patch），并进行维度转置
            unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)       # 1, 25, 100
            # 对展开后的图像小块进行线性变换（分辨率调整），以适应特征图的尺度
            unfold_img = self.resolution_trans(unfold_img)

            # ————————————————————对应论文Fig.3中的2、4步骤————————————————————
            # 对特征图的第 i 个通道进行相同的展开操作
            unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])       # 1, 100, 25
            # 对特征图小块进行分辨率调整，并进行转置，保证它与展开后的图像块对齐
            unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

            # 计算图像小块和特征图小块之间的注意力（通过矩阵乘法）
            # 使用 patch 的面积（patch_size * patch_size）进行归一化
            att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)     # 1, 25, 25

            # 将注意力张量添加一个新的维度，以适应后续的拼接操作
            att = torch.unsqueeze(att, 1)       # 1, 1, 25, 25

            # 将当前通道的注意力结果添加到 attentions 列表中
            attentions.append(att)

        # 将所有通道的注意力张量在维度 1 上进行拼接，形成最终的注意力矩阵
        attentions = torch.cat(attentions, dim=1)       # 1, 12, 25, 25

        return attentions


class AFMA(nn.Module):
    """
        函数功能对应论文Fig.3 (b)
    """
    def __init__(self, out_channels, patch_size=10):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels

        # 同上，用于将特征图分解成patch
        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size),
                                stride=(self.patch_size, self.patch_size))

    def forward(self, x, attentions):

        # 创建一个用于处理特征图大小的卷积核，避免对注意力进行卷积时改变通道数
        conv_feamap_size = nn.Conv2d(self.out_channels, self.out_channels,
                                     kernel_size=(1, 1),
                                     stride=1,
                                     groups=self.out_channels)
        # 将卷积权重初始化为1，并设置为不参加反向传播
        conv_feamap_size.weight = nn.Parameter(
            torch.ones((self.out_channels, 1, 1, 1)))
        conv_feamap_size.to(x.device)
        for param in conv_feamap_size.parameters():
            param.requires_grad = False

        # 创建用于将展开后的特征图重新折叠成原始尺寸的层
        fold_layer = torch.nn.Fold(output_size=(x.size()[-2], x.size()[-1]),
                                   kernel_size=(self.patch_size, self.patch_size),
                                   stride=(self.patch_size, self.patch_size))       # 56, 56, (10, 10), (10, 10)

        correction = []     # 存放与注意力相乘以后的特征图

        x_argmax = torch.argmax(x, dim=1)   # 通过 argmax 找到每个像素的类别

        # 创建 one-hot 编码的特征图
        pr_temp = torch.zeros(x.size()).to(x.device)        # 1, 2, 56, 56
        src = torch.ones(x.size()).to(x.device)             # 1, 2, 56, 56
        x_softmax = pr_temp.scatter(dim=1, index=x_argmax.unsqueeze(1), src=src)    # 1, 2, 56, 56
        # 处理特征图的大小
        argx_feamap = conv_feamap_size(x_softmax) / (2 ** 1 * 2 ** 1)       # 1, 2, 56, 56

        for i in range(x.size()[1]):
            # 计算每个通道的非零注意力值
            non_zeros = torch.unsqueeze(torch.count_nonzero(attentions[:, i:i + 1, :, :], dim=-1) + 0.00001, dim=-1)    # 1, 1, 25, 1

            # 对注意力进行归一化处理，并与展开后的特征图进行矩阵乘法，得到修正后的注意力
            att = torch.matmul(attentions[:, i:i + 1, :, :] / non_zeros,        # 1, 1, 25, 25
                               torch.unsqueeze(self.unfold(argx_feamap[:, i:i + 1, :, :]),      # 1, 100, 25
                                               dim=1                    # 1, 1, 100, 25
                                               ).transpose(-1, -2)      # 1, 1, 25, 100
                               )
            # print(f"attention matrix shape after matmul for channel {i}: {att.shape}")

            att = torch.squeeze(att, dim=1)     # 1, 25, 100
            # print(f"shape before fold for channel {i}: {att.transpose(-1, -2).shape}")

            att = fold_layer(
                att.transpose(-1, -2)   # 1, 100, 25
            )     # 1, 1, 56, 56

            correction.append(att)      # 1, 2, 56, 56

        correction = torch.cat(correction, dim=1)

        x = correction * x + x

        return x, attentions


if __name__ == '__main__':
    x = torch.randn(1, 32, 56, 56)      # 模拟backbone输出的最低维特征
    y = torch.randn(1, 2, 56, 56)       # 模拟decode输出的分割结果

    attention = EncoderChannels(in_channels=32)
    attentions = attention(x)

    AMFA = AFMA(out_channels=2)
    out, _ = AMFA(y, attentions)

    print(out.shape)