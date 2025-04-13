import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelWiseConv(nn.Module):
    def __init__(self, kernel_size, in_channels=3, out_channels=1, stride=1, padding=0):
        super(ChannelWiseConv, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 每个输入通道有自己的一组卷积核
        self.conv = nn.Conv2d(1, out_channels, kernel_size, stride, padding)

    def forward(self, x, x_d):
        x = self.gap(x)
        x_d = x_d *x
        # 对每个通道进行卷积操作
        outputs = torch.zeros_like(x_d)
        for i in range(self.in_channels):

            single_channel = x_d[:, i:i + 1, :, :]  # 取出单个通道
            conv = self.conv(single_channel)
            outputs[:, i:i + 1, :, :] = conv
        # 将各个通道的输出拼接在一起
        return outputs


# 测试通道级卷积层
input_tensor = torch.randn(2, 3, 512, 512)  # 输入张量
channel_wise_conv = ChannelWiseConv(kernel_size=3, padding=1)
output = channel_wise_conv(input_tensor, input_tensor)
print(output.shape)  # 输出张量形状
