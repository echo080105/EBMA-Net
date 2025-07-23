# EBMA-Net
## Introduction
Accurate segmentation of pear leaf diseases is paramount for enhancing diag-nostic precision and optimizing agricultural disease management. However,variations in disease color, texture, and morphology, coupled with changes inlighting conditions and gradual disease progression, pose significant challenges.To address these issues, we propose EBMA-Net, an edge-aware multi-scalenetwork.

EBMA-Net introduces a Multi-Dimensional Joint Attention Module(MDJA) that leverages atrous convolutions to capture lesion information atdifferent scales, enhancing the model’s receptive field and multi-scale process-ing capabilities. 

An Edge Feature Extraction Branch (EFFB) is also designedto extract and integrate edge features, guiding the network’s focus towardsedge information and reducing information redundancy. 

Experiments on a self-constructed pear leaf disease dataset demonstrate that EBMA-Net achieves aMean Intersection over Union (MIoU) of 86.25%, Mean Pixel Accuracy (MPA)of 91.68%, and Dice coeﬀicient of 92.43%, significantly outperforming compari-son models. These results highlight EBMA-Net’s effectiveness in precise pear leafdisease segmentation under complex conditions

## Requirements
Requirements are given below.
```
scipy==1.2.1
numpy==1.17.0
matplotlib==3.1.2
opencv_python==4.1.2.30
torch==1.2.0
torchvision==0.4.0
tqdm==4.60.0
Pillow==8.2.0
h5py==2.10.0
```

## Datasets
Data should be requested to be obtained from the author(22115860@stu.ahau.edu.cn).
```
1. This article uses the VOC format for training.
2. Before training, place the label files in the SegmentationClass folder under the VOC2007 folder within the VOCdevkit folder.
3. Before training, place the image files in the JPEGImages folder under the VOC2007 folder within the VOCdevkit folder.
4. Before training, use the voc_annotation.py file to generate the corresponding txt files.
```
## Training
- Use the below command for training(Note to modify the num_classes in train.py to the number of categories + 1.):
```
python train.py 
```
## Testing
- In the unet.py file, modify the model_path, backbone and num_classes in the following part to make them correspond to the trained files; **model_path refers to the weight file under the logs folder**.    
- Use the below command for testing(In the predict.py file, settings can be made for fps testing and video detection.):
```
python predict.py  
```
## Evaluation Steps
- Set the value of num_classes in get_miou.py to be one more than the number of predicted classes.
- Set the value of name_classes in get_miou.py to the categories that need to be distinguished.
- Run get_miou.py to obtain the miou value.

## Send us feedback
- If you have any queries or feedback, please contact us @(**22115860@stu.ahau.edu.cn**).
