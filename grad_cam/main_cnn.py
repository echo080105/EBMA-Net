import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from grad_cam.utils import GradCAM, show_cam_on_image
from nets.unet import Unet
import numpy as np
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        return logits


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


def main():
    torch.cuda.empty_cache()  # 释放显存
    img_path = r"./img_3.png"
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    rgb_img = np.float32(img) / 255

    data_transform = transforms.Compose([transforms.ToTensor()])

    input_tensor = data_transform(rgb_img)
    input_tensor = torch.unsqueeze(input_tensor, dim=0)
    model = Unet(num_classes=4)
    model.load_state_dict(torch.load(r'D:\something\run\net\分割\ERSA-net\logs\over\best_epoch_weights.pth'))

    # 使用所有预测头作为目标层
    # target_layers = [model.out_head1, model.out_head2, model.out_head3, model.out_head4]
    target_layers = [model]
    model = model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    model = SegmentationModelOutputWrapper(model)
    # from torch.cuda.amp import autocast
    # with autocast():
    output = model(input_tensor)

    logits = output
    normalized_masks = torch.nn.functional.softmax(logits, dim=1).cpu()

    sem_classes = {'background', 'Rust', 'slug', 'crul'}
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

    car_category = sem_class_to_idx["crul"]
    car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
    car_mask_float = np.float32(car_mask == car_category)

    both_images = np.hstack((img, np.repeat(car_mask_uint8[:, :, None], 3, axis=-1)))
    Image.fromarray(both_images)
    plt.imshow(both_images)
    plt.show()

    targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
    with GradCAM(model=model,
                 target_layers=target_layers,
                 use_cuda=torch.cuda.is_available()) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    Image.fromarray(cam_image)

    plt.imshow(cam_image)
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])  # 确保图像填充整个窗口
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 无边距
    plt.margins(0, 0)  # 进一步减少左右边距
    plt.gcf().set_size_inches(plt.gcf().get_size_inches()[0], plt.gcf().get_size_inches()[0], forward=True)  # 保持图像正方形
    plt.show()


if __name__ == '__main__':
    main()



