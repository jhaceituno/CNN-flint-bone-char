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

> The quantification and identification of components in archaeological micromorphology remain subjective and challenging, particularly for early-career researchers. To address this, we developed a deep learning tool for the automatic segmentation of three materials commonly found in Palaeolithic thin sections: bone, charcoal, and flint (including obsidian). Using high-resolution photomicrographs of 57 thin sections in plane-polarised and cross-polarised light, we trained and evaluated state-of-the-art convolutional neural networks (CNNs) for material segmentation. The best-performing configuration, a U-Net with an InceptionV4 encoder, achieved mean intersection over union (IoU) scores of 0.96 for flint (including obsidian), 0.80 for bone, and 0.82 for charcoal. The models also classified the relative abundance of each material with balanced accuracies of 0.99 for flint (including obsidian), 0.92 for bone, and 0.85 for charcoal. These results demonstrate the potential of deep learning to enhance objectivity, accuracy, and reproducibility in archaeological micromorphology, providing a valuable resource for future geoarchaeological research.

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

## Dataset
- [Bone](https://drive.google.com/file/d/1evExk5R5TEgWtITx8Yf1usjC_a_9p3VU/view?usp=drive_link)
- [Charcoal](https://drive.google.com/file/d/1yzW_xC4rvc-ZXg6N4Y2BXV_BXtarSmOJ/view?usp=drive_link)
- [Flint](https://drive.google.com/file/d/1DOxd26lkbXOFSXDH1KeOPKKzH9K0vrYX/view?usp=drive_link)
