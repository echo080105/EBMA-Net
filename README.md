# EBMA-Net
## Introduction
Accurate segmentation of pear leaf diseases is paramount for enhancing diag-
nostic precision and optimizing agricultural disease management. However,
variations in disease color, texture, and morphology, coupled with changes in
lighting conditions and gradual disease progression, pose significant challenges.
To address these issues, we propose EBMA-Net, an edge-aware multi-scale
network. EBMA-Net introduces a Multi-Dimensional Joint Attention Module
(MDJA) that leverages atrous convolutions to capture lesion information at
different scales, enhancing the model’s receptive field and multi-scale process-
ing capabilities. An Edge Feature Extraction Branch (EFFB) is also designed
to extract and integrate edge features, guiding the network’s focus towards
edge information and reducing information redundancy. Experiments on a self-
constructed pear leaf disease dataset demonstrate that EBMA-Net achieves a
Mean Intersection over Union (MIoU) of 86.25%, Mean Pixel Accuracy (MPA)
of 91.68%, and Dice coeﬀicient of 92.43%, significantly outperforming compari-
son models. These results highlight EBMA-Net’s effectiveness in precise pear leaf
disease segmentation under complex conditions
## Requirements


## Training
```
1. This article uses the VOC format for training.
2. Before training, place the label files in the SegmentationClass folder under the VOC2007 folder within the VOCdevkit folder.
3. Before training, place the image files in the JPEGImages folder under the VOC2007 folder within the VOCdevkit folder.
4. Before training, use the voc_annotation.py file to generate the corresponding txt files.
5. Note to modify the num_classes in train.py to the number of categories + 1.
6. Run train.py to start the training.
```

## Citation
```
@article{shu2024enhanced,
  title={Enhanced Disease Segmentation in Pear Leaves via Edge-Aware Multi-Scale Attention Network},
  author={Shu, Xin and Ding, Jie and Wang, Wenwu and Xu, Wenwen and Jiao, Yuxuan and Wu, Yunzhi},
  journal={The Visual Computer},
  year={2024}
}
