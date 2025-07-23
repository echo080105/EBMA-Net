from typing import Dict
import torch.nn.functional as F
import torch
import torch.nn as nn


class ConvBNReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(ConvBNReLUBlock, self).__init__()
        self.size = size
        self.conv_bn_silu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=size, padding=(size - 1) // 2),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv_bn_silu(x)
        return out


class TQ(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TQ, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_a = x
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_4 = self.conv4(x)
        return out


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        return x * self.weight + self.bias


class GMSA(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False
                 ):
        super().__init__()

        self.gn = GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()
        self.tq = TQ(oup_channels, oup_channels)

    def forward(self, x):
        gn_x = self.gn(x)
        x = x * reweigts
        out = self.tq(x)

        return out


class AGC(nn.Module):
    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        # up
        self.GWC = nn.Sequential(
            # GCT(up_channel // squeeze_radio),
            nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                      padding=group_kernel_size // 2, groups=group_size)
        )
        self.PWC1 = nn.Sequential(
            # GCT(up_channel // squeeze_radio),
            nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        )
        # low
        self.PWC2 = nn.Sequential(
            # GCT(low_channel // squeeze_radio),
            nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                      bias=False)
        )
        self.advavg = nn.AdaptiveAvgPool2d(1)
        # self.channelattention = ChannelAttention(op_channel, op_channel // 4)




class MDJA(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.GMSA = GMSA(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.AGC = AGC(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)
        # self.conv = nn.Conv2d(op_channel * 2, op_channel, kernel_size=1)




class EFE(nn.Module):
    def __init__(self, in_channels, sobel_strength=1):
        super(EFE, self).__init__()
        self.sobel_strength = sobel_strength
        self.in_channels = in_channels
        # self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        sobel_x = sobel_strength * torch.tensor([[-1, 0, 1],
                                                 [-2, 0, 2],
                                                 [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_strength * torch.tensor([[-1, -2, -1],
                                                 [0, 0, 0],
                                                 [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel = sobel_strength * torch.tensor([[-1, -1, -1],
                                               [-1, 9, -1],
                                               [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)


        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('sobel', sobel)
        self.bn = nn.BatchNorm2d(in_channels)
        self.silu = nn.SiLU(inplace=True)

        self.sigmoid = nn.Sigmoid()




class BFA(nn.Module):
    def __init__(self, in_channels, r, size):
        super(BFA, self).__init__()

        self.in_channel = in_channels
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.convo = ConvBNReLUBlock(in_channels * 2, in_channels, 1)
        self.conv4_1 = nn.Sequential(
            # GCT(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(4, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1_4 = nn.Sequential(
            # GCT(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 4)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.convo_r1 = ConvBNReLUBlock(in_channels * 2, in_channels * 2 // r, 1)
        self.conv3 = ConvBNReLUBlock(in_channels * 2 // r, in_channels * 2 // r, 3)
        self.convo_r2 = ConvBNReLUBlock(in_channels * 2 // r, in_channels, 1)
        self.tq = TQ(1, in_channels)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = ConvBNReLUBlock(1, in_channels, 1)
        # self.gate = GCT(1)
        self.convp = MDJA(in_channels)

    def forward(self, x, edge):
        x_a = self.gap(x)
        x_m = self.gmp(x)
        # edge = self.tq(edge)
        # print(x.shape, edge.shape)
        # edge = self.gate(edge)
        edge = self.conv1(edge)
        # edge = self.convp(edge)

        e_a = self.gap(edge)
        e_m = self.gmp(edge)
        w_1 = self.sigmoid(self.conv4_1(torch.cat((x_a, e_a, x_m, e_m), dim=2)))
        w_2 = self.sigmoid(self.conv1_4(torch.cat((x_a, e_a, x_m, e_m), dim=3)))
        x_r = x + edge * w_1
        e_r = edge + x * w_2
        o = torch.cat((x_r, e_r), dim=1)
        output = self.convo(o) + self.convo_r2(self.conv3(self.convo_r1(o)))
        return output


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_prob=0.):
        super(DoubleConv, self).__init__()
        # self.gate = GCT(in_channels)
        # mid_channels = (in_channels + out_channels) // 2
        # self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)  # 第一个卷积层
        # self.bn1 = nn.BatchNorm2d(mid_channels)  # 批标准化层
        # # nn.ReLU(inplace=True),  # ReLU 激活函数
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)  # 第二个卷积层
        self.bn = nn.BatchNorm2d(out_channels)  # 批标准化层
        self.silu = nn.SiLU(inplace=True)  # ReLU 激活函数

    def forward(self, x):
        # x = self.gate(x)
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.silu(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.convp = MDJA(in_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层进行下采样
        self.convdown = DoubleConv(in_channels, out_channels)  # 双卷积块
        # self.gate = GCT(in_channels)

    def forward(self, x):
        x_c = self.convp(x)
        x = x + x_c
        x = self.maxpool(x)
        x = self.convdown(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.sigmoid = nn.Sigmoid()
        # self.conv = nn.Conv2d(1, 1, 7, padding=3)
        # self.convl = DoubleConv(in_channels, out_channels)  # 双卷积块
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.convp = MDJA(out_channels)

        # self.conv_rh = ScConv_up(in_channels)
        # if bilinear:
        #       # 双线性插值上采样
        self.conv = DoubleConv(in_channels * 2, out_channels, in_channels // 2)  # 双卷积块
        # else:
        #     self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  # 转置卷积进行上采样
        #

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # out = self.conv_rh(x1, x2)
        # out = self.convl(out)

        # print(x1.shape)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        # print(x.shape)
        out = self.conv(x)

        return out


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1),  # 1x1 卷积层
            # nn.BatchNorm2d(num_classes),  # 批标准化层
            # nn.ReLU(inplace=True)  # ReLU 激活函数
        )


class Unet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 in_channels: int = 3,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.efe = EFE(in_channels)
        self.in_conv = DoubleConv(in_channels, base_c)  # 输入的双卷积块
        self.down1 = Down(base_c, base_c * 2)  # 下采样层 1
        self.down2 = Down(base_c * 2, base_c * 4)  # 下采样层 2
        self.down3 = Down(base_c * 4, base_c * 8)  # 下采样层 3
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)  # 下采样层 4

        #      self.down = encoder()
        self.convo = ConvBNReLUBlock(base_c * 16, base_c * 8, 1)
        self.up1 = Up(base_c * 8, base_c * 8 // factor, bilinear)  # 上采样层 1
        self.up2 = Up(base_c * 4, base_c * 4 // factor, bilinear)  # 上采样层 2
        self.up3 = Up(base_c * 2, base_c * 2 // factor, bilinear)  # 上采样层 3
        self.up4 = Up(base_c, base_c, bilinear)  # 上采样层 4
        self.bfa = BFA(base_c, 4, 1)
        self.out_conv = OutConv(base_c, num_classes)  # 输出卷积层

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_e = self.efe(x)
        x1 = self.in_conv(x)  # 输入双卷积块
        x2 = self.down1(x1)  # 下采样层 1
        x3 = self.down2(x2)  # 下采样层 2
        x4 = self.down3(x3)  # 下采样层 3
        x5 = self.down4(x4)  # 下采样层 4

        # print(x1.shape, x2.shape, x3.shape, x4.shape,x5.shape)
        x = self.up1(x5, x4)  # 上采样层 1
        x = self.up2(x, x3)  # 上采样层 2
        x = self.up3(x, x2)  # 上采样层 3
        x = self.up4(x, x1)  # 上采样层 4

        x = self.bfa(x, x_e)
        logits = self.out_conv(x)  # 输出卷积层

        return logits


import time

if __name__ == "__main__":
    device = 'cpu'

    start_time = time.time()

    # 创建一个随机输入张量，形状为 (batch_size, channels, height, width)
    img_tensor = torch.randn(2, 3, 512, 512).to(device)  # 假设批量大小为 2，通道数为 3，高度和宽度均为 512

    model_init_start = time.time()
    model = Unet(num_classes=4).to(device)  # 实例化 Unet 模型并移动到GPU
    model_init_end = time.time()

    forward_start = time.time()
    # 前向传播
    output_tensor = model(img_tensor)
    forward_end = time.time()

    end_time = time.time()

    # 打印输出张量的形状
    print("输出张量的形状:", output_tensor.shape)
    print("总耗时: {:.2f} 秒".format(end_time - start_time))
    print("模型初始化时间: {:.2f} 秒".format(model_init_end - model_init_start))
    print("前向传播时间: {:.2f} 秒".format(forward_end - forward_start))
