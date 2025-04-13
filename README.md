1. This article uses the VOC format for training.
2. Before training, place the label files in the SegmentationClass folder under the VOC2007 folder within the VOCdevkit folder.
3. Before training, place the image files in the JPEGImages folder under the VOC2007 folder within the VOCdevkit folder.
4. Before training, use the voc_annotation.py file to generate the corresponding txt files.
5. Note to modify the num_classes in train.py to the number of categories + 1.
6. Run train.py to start the training.