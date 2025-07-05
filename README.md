## Improving Micromorphological Analysis with CNN-Based Segmentation of Flint/Obsidian, Bone, and Charcoal

Presented at *Computers & Geosciences*

- Version: 1.0
- License: MIT

## Authors

- Rafael Arnay-del Arco
- Pedro García-Villa
- Javier Hernández-Aceituno
- Sara Rueda-Saiz
- Carolina Mallol  

## Abstract

> The quantification and identification of components in archaeological micromorphology remain subjective and challenging, particularly for early-career researchers. To address this, we developed a deep learning-based tool for the automatic segmentation of three key materials frequently found in Paleolithic thin sections: bone, charcoal, and flint/obsidian. Leveraging high-resolution photomicrographs and convolutional neural network architectures, including U-Net, FPN, and PSPNet, combined with various pre-trained encoders, we trained and evaluated models capable of precise material segmentation. The U-Net architecture with the InceptionV4 encoder achieved the best performance, particularly for flint/obsidian, with a dataset-level IoU of 0.96. The models also performed well in classifying the relative abundance of each material, achieving high balanced accuracy across categories despite dataset imbalance. Our results demonstrate the potential of deep learning to improve objectivity, accuracy, and reproducibility in archaeological micromorphology, providing a valuable tool for future geoarchaeological research.

## Manual

The presented code expects to be run within Python3 cell of a Google Colaboratory Jupyter Notebook, although it could be easily adjusted to run as an independent Python script by modifying the file access and installation lines.

Before running the code, the following global variables should be adjusted:

- `USER` (*str*): User name to allow for concurrent training. If the script is not expected to be executed by several users at once, this value is irrelevant.
- `TARGET_CLASS` (*str*): Target class which the model will be trained to detect. It should be `'flint'`, `'charcoal'` or `'bone'`.
- `DATASET_NAME` (*str*): Name of the `rar` file or folder where the training dataset is located
- `COLAB_PATH` (*str*): Path to the root folder where all training files are located and results should be saved
- `DATE_DIR_NAME` (*str*): Name of specific training folder if one should be used. If `None`, the script will automatically search most recent one available for the current values of `USER`, `TARGET_CLASS`, `ARCH` and `ENCODER`.
- `TRAIN` (*str*): `True` if a model must be trained, `False` if it must only be tested.
- `ARCH` (*str*): Model architecture to be used. In the presented work, only `'Unet'`, `'FPN'`, and `'PSPNet'` were used.
- `ENCODER` (*str*): Model encoder to be used. In the presented work, only `'resnet50'`, `'vgg19'`, `'inceptionv4'` and `'xception'` were used
- `MAX_EPOCHS` (*int*): Maximum number of training epochs.
- `BATCH_SIZE` (*int*): Training batch size.
- `TRAIN_SIZE` (*float*): Percentage of samples to be used for traning (as a fraction of 1.0).
- `VAL_SIZE` (*float*): Percentage of samples to be used for validation (as a fraction of 1.0).