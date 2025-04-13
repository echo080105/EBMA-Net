from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, \
    XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2
import numpy as np
import torch
from nets.unet import Unet

#离线模型,模型定义具体省略
my_model = Unet(num_classes=4)  #省略
model_pkl = "../logs/over/best_epoch_weights.pth"  #加载自己训练好的模型
my_model.load_state_dict(torch.load(model_pkl))
my_model.eval()

# 判断是否使用 GPU 加速
use_cuda = torch.cuda.is_available()
if use_cuda:
    my_model = my_model.cuda()  #如果是gpu的话加速


# 首先定义函数对vit输出的3维张量转换为传统卷积处理时的二维张量，gradcam需要。
# （B,H*W,feat_dim）转换到（B,C,H,W）,其中H*W是分pathc数。具体参数根据自己模型情况
# 我的输入为224*224，pathsize为（16*16），那么我的（H，W）就是(224/16，224/16)，即14*14
def reshape_transform(tensor, height=14, width=14):
    # 去掉cls token
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result


# 创建 GradCAM 对象
cam = GradCAM(model=my_model,
              target_layers=[my_model.out_conv],
              # 这里的target_layer要看模型情况，调试时自己打印下model吧
              # 比如还有可能是：target_layers = [model.blocks[-1].ffn.norm]
              # 或者target_layers = [model.blocks[-1].ffn.norm]

              reshape_transform=reshape_transform)

# 读取输入图像
image_path = "./Rust_278.jpg"
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (512, 512))

# 预处理图像
input_tensor = preprocess_image(rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

# 看情况将图像转换为批量形式
# input_tensor = input_tensor.unsqueeze(0)
if use_cuda:
    input_tensor = input_tensor.cuda()

# 计算 grad-cam
target_category = None  # 可以指定一个类别，或者使用 None 表示最高概率的类别
grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)
grayscale_cam = grayscale_cam[0, :]

# 将 grad-cam 的输出叠加到原始图像上
#visualization = show_cam_on_image(rgb_img, grayscale_cam)，借鉴的代码rgb格式不对，换成下面
visualization = show_cam_on_image(rgb_img.astype(dtype=np.float32) / 255, grayscale_cam)

# 保存可视化结果
cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)  ##注意自己图像格式，吐过本身就是BGR，就不用这行

cv2.imwrite('cam.jpg', visualization)
