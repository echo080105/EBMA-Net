import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# class densecat_cat_add(nn.Module):
#     def __init__(self, in_chn, out_chn):
#         super(densecat_cat_add, self).__init__()
#
#         self.conv1 = torch.nn.Sequential(
#             torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
#             torch.nn.ReLU(inplace=True),
#         )
#         self.conv2 = torch.nn.Sequential(
#             torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
#             torch.nn.ReLU(inplace=True),
#         )
#         self.conv3 = torch.nn.Sequential(
#             torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
#             torch.nn.ReLU(inplace=True),
#         )
#         self.conv_out = torch.nn.Sequential(
#             torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
#             torch.nn.BatchNorm2d(out_chn),
#             torch.nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x, y):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x2 + x1)
#
#         y1 = self.conv1(y)
#         y2 = self.conv2(y1)
#         y3 = self.conv3(y2 + y1)
#
#         return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)
#
#
# class densecat_cat_diff(nn.Module):
#     def __init__(self, in_chn, out_chn):
#         super(densecat_cat_diff, self).__init__()
#         self.conv1 = torch.nn.Sequential(
#             torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
#             torch.nn.ReLU(inplace=True),
#         )
#         self.conv2 = torch.nn.Sequential(
#             torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
#             torch.nn.ReLU(inplace=True),
#         )
#         self.conv3 = torch.nn.Sequential(
#             torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
#             torch.nn.ReLU(inplace=True),
#         )
#         self.conv_out = torch.nn.Sequential(
#             torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
#             torch.nn.BatchNorm2d(out_chn),
#             torch.nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x, y):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x2 + x1)
#
#         y1 = self.conv1(y)
#         y2 = self.conv2(y1)
#         y3 = self.conv3(y2 + y1)
#         out = self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))
#         return out
#
#
# class DF_Module(nn.Module):
#     def __init__(self, dim_in, dim_out, reduction=True):
#         super(DF_Module, self).__init__()
#         if reduction:
#             self.reduction = torch.nn.Sequential(
#                 torch.nn.Conv2d(dim_in, dim_in // 2, kernel_size=1, padding=0),
#                 nn.BatchNorm2d(dim_in // 2),
#                 torch.nn.ReLU(inplace=True),
#             )
#             dim_in = dim_in // 2
#         else:
#             self.reduction = None
#         self.cat1 = densecat_cat_add(dim_in, dim_out)
#         self.cat2 = densecat_cat_diff(dim_in, dim_out)
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(dim_out),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x1, x2):
#         if self.reduction is not None:
#             x1 = self.reduction(x1)
#             x2 = self.reduction(x2)
#         x_add = self.cat1(x1, x2)
#         x_diff = self.cat2(x1, x2)
#         y = self.conv1(x_diff) + x_add
#         return y


# 定义 Edge 类


import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

# 边缘检测类
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(input_dim // 3, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CrossConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossConv, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv3_1_A = nn.Conv2d(in_channels, out_channels, (3, 1), padding=(1, 0))
        self.conv3_1_B = nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1))
        self.relu_1 = nn.LeakyReLU(inplace=True)
        self.conv3_2_A = nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0))
        self.conv3_2_B = nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1))
        self.adafm = nn.Conv2d(out_channels, out_channels, 3, padding=3 // 2, groups=out_channels)
        self.senet = SEBlock(out_channels)

    def forward(self, input, other):
        other = self.gap(other)
        input = input * other
        x = self.conv3_1_A(input) + self.conv3_1_B(input)
        x = self.relu_1(x)
        x = self.conv3_2_A(x) + self.conv3_2_B(x)
        x = self.adafm(x) + x
        x = self.senet(x) * input
        return x


class RH(nn.Module):
    def __init__(self, in_channels):
        super(RH, self).__init__()
        self.out_channels = in_channels
        self.cb1 = ConvBlock(in_channels * 2, self.out_channels, 1, 0)
        self.cb2 = ConvBlock(in_channels * 2, self.out_channels, 3, 1)
        self.cb3 = ConvBlock(in_channels * 2, self.out_channels, 5, 2)
        self.cross = CrossConv(self.out_channels, self.out_channels)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x_1 = self.cb1(x)
        x_2 = self.cb2(x)
        x_3 = self.cb3(x)

        x_all = x_1 + x_2 + x_3

        other = self.cb1(x)

        x_all = x_all * self.gap(other)

        return x_all


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(ConvBNReLUBlock, self).__init__()
        self.size = size
        self.conv_bn_relu_7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_bn_relu_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.size == 7:
            return self.conv_bn_relu_7(x)
        elif self.size == 1:
            return self.conv_bn_relu_1(x)


class TQ(nn.Module):
    def __init__(self, size, out_channels):
        super(TQ, self).__init__()
        # 然后应用最大池化层，将尺寸缩小 16 倍
        self.max_pool = nn.MaxPool2d(kernel_size=size, stride=size)

        self.conv1 = ConvBNReLUBlock(1, 2, 7)
        self.conv2 = ConvBNReLUBlock(2, 4, 7)
        self.conv3 = ConvBNReLUBlock(4, 8, 7)
        self.conv4 = ConvBNReLUBlock(8, 4, 7)
        self.reconv = ConvBNReLUBlock(4, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x_1.shape)
        w = self.max_pool(x)
        w = self.conv1(w)
        w = self.conv2(w)
        w = self.conv3(w)
        w = self.conv4(w)
        w = self.reconv(w)
        w = self.sigmoid(w)
        # print(w.shape)
        return w * x


class Edge(nn.Module):
    def __init__(self, in_channels, sobel_strength=1):
        super(Edge, self).__init__()
        self.sobel_strength = sobel_strength
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        sobel_x = sobel_strength * torch.tensor([[-1, 0, 1],
                                                 [-2, 0, 2],
                                                 [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_strength * torch.tensor([[-1, -2, -1],
                                                 [0, 0, 0],
                                                 [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel = sobel_strength * torch.tensor([[-1, -1, -1],
                                               [-1, 8, -1],
                                               [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)

        # Register buffers to ensure they are moved to the right device with the model
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('sobel', sobel)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.SiLU(inplace=True)
        self.threshold = 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Expand sobel filters to match input channels
        sobel_x = self.sobel_x.expand(self.in_channels, 1, 3, 3)
        sobel_y = self.sobel_y.expand(self.in_channels, 1, 3, 3)
        sobel = self.sobel.expand(self.in_channels, 1, 3, 3)
        #
        x_o = F.conv2d(x, sobel, padding=1, groups=self.in_channels)
        # grad1 = self.bn(x_o)
        # x_o = self.relu(grad1)
        # x_o = x_o.clamp(0, 1)
        y = self.sigmoid(x_o) * x + x
        #
        grad_x = F.conv2d(y, sobel_x, padding=1, groups=self.in_channels)
        grad_y = F.conv2d(y, sobel_y, padding=1, groups=self.in_channels)
        grad2 = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        grad2 = self.bn(grad2)
        grad2 = self.relu(grad2)
        grad2 = grad2.sum(dim=1, keepdim=True)
        # # print(output)
        #
        x_1 = grad2.clamp(0, 1)
        output = self.sigmoid(x_1) * x_1 + x_1

        return output


class RH(nn.Module):
    def __init__(self, in_channels, r):
        super(RH, self).__init__()
        # self.way = way
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

        self.convo = ConvBNReLUBlock(in_channels * 2, in_channels, 1)
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(4, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 4)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1 = ConvBNReLUBlock(1, in_channels, 1)
        self.convd = ConvBNReLUBlock(in_channels, in_channels // r, 1)
        self.convdr = nn.Sequential(
            nn.Conv2d(in_channels // r, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        self.convo_r1 = ConvBNReLUBlock(in_channels * 2, in_channels * 2 // r, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels * 2 // r, in_channels * 2 // r, 3, padding=1),
            nn.BatchNorm2d(in_channels * 2 // r),
            nn.ReLU(inplace=True)
        )
        self.convo_r2 = ConvBNReLUBlock(in_channels * 2 // r, in_channels, 1)
        self.tq_s = TQ(in_channels)
        self.tq_d = TQ(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(in_channels)
        self.convs = ConvBNReLUBlock(2, 1, 7)

    def forward(self, x, edge):
        edge = self.tq_d(edge)
        x = x + x * edge
        x_a = self.gap(x)
        x_m = self.gmp(x)
        x_all = self.convdr(self.convd(x_a)) + self.convdr(self.convd(x_m))
        x_c = x * self.sigmoid(x_all)

        x_m, _ = torch.max(x_c, dim=1, keepdim=True)
        x_a = torch.mean(x_c, dim=1, keepdim=True)
        x_s = self.convs(torch.cat((x_m, x_a), dim=1))
        output = x_c * self.sigmoid(x_s)
        return output


# if __name__ == "__main__":
#     # 创建一个随机输入张量，形状为 (batch_size, channels, height, width)
#     img1 = torch.randn(2, 16, 512, 512)  # 假设批量大小为 2，通道数为 3，高度和宽度均为 512
#     img2 = torch.randn(2, 1, 512, 512)
#     model = (16, 2)  # 实例化 Unet 模型
#
#     # 前向传播
#     output_tensor = model(img1, img2)
#
#     # 打印输出张量的形状
#     print("输出张量的形状:", output_tensor.shape)
# 读取并预处理图片
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image


# 显示图片
def show_image(tensor):
    image = tensor[0, 0].cpu().detach().numpy()
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


class EdgeConv(nn.Module):
    def __init__(self, in_channels):
        super(EdgeConv, self).__init__()
        # 定义3x3边缘卷积核，中心为1，周围为-1
        self.edge_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        # 创建自定义的卷积核权重
        edge_weight = torch.tensor([[-1.0, -1.0, -1.0],
                                    [-1.0, 1.0, -1.0],
                                    [-1.0, -1.0, -1.0]])
        # 将该权重扩展到所有输入通道
        self.edge_conv.weight = nn.Parameter(edge_weight.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1))

    def forward(self, x):
        return self.edge_conv(x)


# 主函数
def main(image_path):
    # 读取图片
    image = load_image(image_path)

    # 打印原始图片
    print("Original Image:")
    show_image(image)

    # 定义边缘检测层
    edge_detection_layer1 = Edge(3)
    edge_detection_layer2 = Edge(1)
    model = EdgeConv(3)
    # 执行边缘检测
    edges = edge_detection_layer1(image)
    # edges = edge_detection_layer2(edges)
    # edges = model(image)
    # edges = model(edges)
    print(edges.shape)
    # 打印边缘检测后的图片
    print("Edges Detected:")
    show_image(edges)
    # print(edges)


# 使用图片路径
image_path = r'D:\something\run\net\分割\ERSA-net\img\Rust_f7.jpg'
main(image_path)

# import torch
# import torch.nn as nn
#

#
# if __name__ == '__main__':
#     from thop import profile
#
#     model = DF_Module(512, 512, True)
#     x1 = torch.randn(2, 512, 32, 32)
#     x2 = torch.randn(2, 512, 32, 32)
#     y = model(x1, x2)
#     print(y.shape)

# import numpy as np
# from sklearn.manifold import TSNE
#
# """将3维数据降维2维"""
#
# # 4个3维的数据
# x = np.array([[0, 0, 0, 1, 2], [0, 1, 1, 3, 5], [1, 0, 1, 7, 2], [1, 1, 1, 10, 22]])
# # 嵌入空间的维度为2，即将数据降维成2维
# ts = TSNE(perplexity=2, n_components=3)
# # 训练模型
# ts.fit_transform(x)
# # 打印结果
# print(ts.embedding_)
