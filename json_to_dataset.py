import PIL.Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import base64
import json
import os
import os.path as osp

import numpy as np

from labelme import utils


if __name__ == '__main__':
   
    jpgs_path = "./Rust(1)/rustf"
    
    pngs_path = "./Rust(1)/rustf_mask"
    
    classes = ["_background_", "Rust", "curl", "slug"]
    
    count = os.listdir("./Rust(1)/Rust_f/")
    # 遍历文件列表
    for i in range(0, len(count)):
        # 获取当前文件的完整路径
        path = os.path.join("./Rust(1)/Rust_f", count[i])
        # 检查当前路径是否指向json文件
        if os.path.isfile(path) and path.endswith('json'):
            # 加载json文件
            data = json.load(open(path))
            # 检查json数据中是否包含图像数据，如果包含就直接使用，否则从文件中加载
            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')
            # 将图像数据从base64编码转换为图像数组
            img = utils.img_b64_to_arr(imageData)
            # 初始化标签名称到标签值的映射
            label_name_to_value = {'_background_': 0}
            # 从shapes字段中获取标签名称，并更新标签名称到标签值的映射
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            
            # 确保标签值是紧凑的，即没有跳过任何数字
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))
            # 根据shapes字段，将标签应用到图像上，生成一个标签数组
            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
            # 将原始图像保存为jpg
            PIL.Image.fromarray(img).save(osp.join(jpgs_path, count[i].split(".")[0]+'.jpg'))
            # 初始化新的分割掩码
            new = np.zeros([np.shape(img)[0], np.shape(img)[1]])
            for name in label_names:
                index_json = label_names.index(name)
                index_all = classes.index(name)
                # 更新掩码，将标签值应用到对应的像素上
                new = new + index_all*(np.array(lbl) == index_json)

            utils.lblsave(osp.join(pngs_path, count[i].split(".")[0]+'.png'), new)
            print('Saved ' + count[i].split(".")[0] + '.jpg and ' + count[i].split(".")[0] + '.png')
