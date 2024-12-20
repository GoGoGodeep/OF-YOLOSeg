import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck



class AttentionBasedSplitC2F(nn.Module):
    def __init__(self, channels, reduction=16):
        super(AttentionBasedSplitC2F, self).__init__()
        self.channels = channels
        self.reduction = reduction

        # 全局池化和注意力权重生成
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)  # 降维
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)  # 恢复维度
        self.sigmoid = nn.Sigmoid()  # 注意力权重通过 Sigmoid 激活

    def forward(self, x):       # 1, 512, 14, 14
        b, c, _, _ = x.size()   # b=512, c=1

        # Step 1: 计算每个通道的全局特征向量
        y = self.global_avg_pool(x).view(b, c)  # (b, c, 1, 1) -> (b, c)    # 1, 512

        # Step 2: 通过全连接层生成注意力权重
        y = self.fc1(y)  # 降维       # 1, 512//16=32
        y = F.relu(y, inplace=True)
        y = self.fc2(y)  # 恢复原通道数       # 1, 512
        attention_weights = self.sigmoid(y).view(b, c, 1, 1)  # (b, c) -> (b, c, 1, 1)      # 1, 512, 1, 1

        # Step 3: 根据注意力权重拆分特征图
        x_attentioned = x * attention_weights  # 加权输入特征图，基于通道注意力    # 1, 512, 14, 14
        high_attention, low_attention = self.split_based_on_attention(x_attentioned, attention_weights)     # 1, 512, 14, 14

        return high_attention, low_attention

    def split_based_on_attention(self, x, attention_weights):
        # 计算注意力阈值，将特征图分为高注意力和低注意力两部分
        threshold = attention_weights.mean(dim=1, keepdim=True)
        high_attention = x * (attention_weights >= threshold).float()  # 高注意力部分
        low_attention = x * (attention_weights < threshold).float()  # 低注意力部分
        return high_attention, low_attention


class C2f_Attention(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False):    # 128, 256
        super().__init__()
        self.c = int(c2 * 0.5)      # 128
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)       # 128, 256
        self.cv2 = Conv((2 + n) * 2 * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(2 * self.c, 2 * self.c,
                                          shortcut, g=1, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

        self.attention = AttentionBasedSplitC2F(channels=2 * self.c, reduction=16)

    def forward(self, x):
        y1 = self.cv1(x)        # 256
        high_attention, low_attention = self.attention(y1)  # 256, 256
        y2 = self.m[0](high_attention)      # 256
        return self.cv2(torch.cat([low_attention, high_attention, y2], 1))


if __name__ == '__main__':
    img = torch.rand(16, 128, 56, 56)

    attention_split_layer = AttentionBasedSplitC2F(channels=128)
    high_attention, low_attention = attention_split_layer(img)
    print(high_attention.shape)
    print(low_attention.shape)
