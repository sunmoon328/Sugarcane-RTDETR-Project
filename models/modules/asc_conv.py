import torch
import torch.nn as nn


class AsymmetricStripConv(nn.Module):
    """
    非对称条形卷积 (ASC) 模块
    在主干网络深层特征提取阶段，加入 1xN 和 Nx1 的条形卷积核。
    """

    def __init__(self, in_channels, out_channels, N=9):
        super(AsymmetricStripConv, self).__init__()
        padding = N // 2

        # 降维与特征整合
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Nx1 纵向卷积 (增强甘蔗主体的纵向特征)
        self.conv_nx1 = nn.Conv2d(out_channels, out_channels, kernel_size=(N, 1), padding=(padding, 0), bias=False)
        self.bn_nx1 = nn.BatchNorm2d(out_channels)

        # 1xN 横向卷积
        self.conv_1xn = nn.Conv2d(out_channels, out_channels, kernel_size=(1, N), padding=(0, padding), bias=False)
        self.bn_1xn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn_reduce(self.reduce(x)))

        # 两条分支特征相加
        out_nx1 = self.bn_nx1(self.conv_nx1(x))
        out_1xn = self.bn_1xn(self.conv_1xn(x))

        return self.relu(out_nx1 + out_1xn)