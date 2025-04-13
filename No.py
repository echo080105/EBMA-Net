import os
from PIL import Image
import numpy as np


def main():
    img_channels = 3
    img_dir = r"D:\something\Code\run\net\分割\unet\VOCdevkit\VOC2007\JPEGImages"
    roi_dir = r"D:\something\Code\run\net\分割\unet\VOCdevkit\VOC2007\SegmentationClass"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    assert os.path.exists(roi_dir), f"roi dir: '{roi_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".jpg")]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    total_samples = 0
    i = 0
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        ori_path = os.path.join(roi_dir, img_name.replace(".jpg", ".png"))
        img = np.array(Image.open(img_path)) / 255.
        roi_img = np.array(Image.open(ori_path).convert('L'))

        img = img[roi_img == 255]
        cumulative_mean += img.mean(axis=0)
        cumulative_std += img.std(axis=0)
        total_samples += img.shape[0]
        i += 1
        print(i)

    mean = cumulative_mean / total_samples
    std = cumulative_std / total_samples
    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__ == '__main__':
    main()
