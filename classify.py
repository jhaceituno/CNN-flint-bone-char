import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from google.colab import drive
from datetime import datetime
import albumentations as alb
from pprint import pprint
from PIL import Image
import numpy as np
import random
import torch
import glob
import cv2
import os

from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from albumentations.pytorch import ToTensorV2

USER = 'user'  # User name allows for concurrent training
TARGET_CLASS = 'flint'  # flint | charcoal | bone
DATASET_NAME = f'{TARGET_CLASS}_dataset'  # name of the training dataset folder
COLAB_PATH = '/content/gdrive/MyDrive/segmentation/'  # Google Drive root folder

DATE_DIR_NAME = None  # Use specific training folder - will automatically search most recent if None
TRAIN = True   # Set to False when only testing
ARCH = 'Unet'  # Model architecture
ENCODER = 'inceptionv4' # Model encoder

MAX_EPOCHS = 40
BATCH_SIZE = 16
TRAIN_SIZE = 0.7
VAL_SIZE = 0.2

drive.mount('/content/gdrive', force_remount=True)
!pip install segmentation-models-pytorch
!pip install pytorch-lightning

path = f"'{os.path.join(COLAB_PATH, DATASET_NAME)}.rar'"
!unrar x $path


class SedimentsDataset(Dataset):
    CLASSES = ['_background_', 'bone', 'charcoal', 'flint']

    def __init__(self, images_dir, masks_dir, target_class='bone', augmentation=None,
                 preprocessing=None, transform=None, mode='train'):
          
        self.ids = os.listdir(images_dir)
        self.mode = mode
        random.seed(42)

        # Shuffle the data
        data_copy = self.ids[:]
        random.shuffle(data_copy)

        # Determine the split indices
        total_len = len(data_copy)
        train_size = TRAIN_SIZE
        val_size = VAL_SIZE

        train_end = int(total_len * train_size)
        val_end = train_end + int(total_len * val_size)

        # Split the data
        self.ids = data_copy[:train_end] if mode == 'train' else (
                   data_copy[train_end:val_end] if mode == 'val' else data_copy[val_end:])
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_value = self.CLASSES.index(target_class.lower())
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.transform = transform

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_fps[i], 0)
        mask = mask.squeeze()
        mask = mask.astype(np.float32)
        mask[mask != self.class_value] = 0.0
        mask[mask == self.class_value] = 1.0

        image = np.array(Image.fromarray(image).resize((512, 512), Image.BILINEAR))
        mask = np.array(Image.fromarray(mask).resize((512, 512), Image.NEAREST))

        if self.mode == 'train' and self.transform:
          augmented = self.transform(image=image, mask=mask)
          image = augmented['image']
          mask = augmented['mask']

        image = np.moveaxis(image, -1, 0)
        mask = np.expand_dims(mask, 0)
        return {'image': image, 'mask': mask}

    def __len__(self):
        return len(self.ids)


class SedimentsModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, encoder_weights, in_channels, out_classes, **kwargs):
        super().__init__()
        self.train_step_outputs = []
        self.valid_step_outputs = []
        self.test_step_outputs = []
        self.model = smp.create_model(arch, encoder_name=encoder_name, encoder_weights=encoder_weights,
                                      in_channels=in_channels, classes=out_classes, **kwargs)
        self.save_hyperparameters()
        # preprocessing parameters for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer('std', torch.tensor(params['std']).view(1, 3, 1, 1))
        self.register_buffer('mean', torch.tensor(params['mean']).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        # normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch['image']

        # Shape of the image should be (batch_size, num_channels, height, width)
        # with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch['mask']

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode='binary')

        output= {'loss': loss, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
        if stage == 'train':
          self.train_step_outputs.append(output)
        elif stage == 'test':
          self.test_step_outputs.append(output)
        else:
          self.valid_step_outputs.append(output)
        return output

    def shared_epoch_end(self, stage):
        # aggregate step metics
        outputs = self.train_step_outputs if stage == 'train' else (
                  self.test_step_outputs if stage == 'test' else self.valid_step_outputs)

        tp = torch.cat([x['tp'] for x in outputs])
        fp = torch.cat([x['fp'] for x in outputs])
        fn = torch.cat([x['fn'] for x in outputs])
        tn = torch.cat([x['tn'] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro-imagewise')

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with 'empty' images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')

        metrics = {f'{stage}_per_image_iou': per_image_iou, f'{stage}_dataset_iou': dataset_iou}

        self.log_dict(metrics, prog_bar=True)
        # free memory
        if stage == 'train':
          self.train_step_outputs.clear() 
        elif stage == 'test':
          self.test_step_outputs.clear()
        else:
          self.valid_step_outputs.clear()

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')

    def on_train_epoch_end(self):
        return self.shared_epoch_end('train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'valid')

    def on_validation_epoch_end(self):
        return self.shared_epoch_end('valid')

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')

    def on_test_epoch_end(self):
        return self.shared_epoch_end('test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


def get_most_recent_model(folder, extension='.ckpt'):
    files = glob.glob(os.path.join(folder, '**', f'*{extension}'), recursive=True)
    return max(files, key=os.path.getmtime) if files else None

def get_best_model(folder, extension='.ckpt'):
    files = glob.glob(os.path.join(folder, '**', f'best_model-*{extension}'), recursive=True)
    return max(files, key=os.path.getmtime) if files else None


transform = alb.Compose([alb.HorizontalFlip(p=0.5), alb.VerticalFlip(p=0.5)])
data_dir = os.path.join('/content/', DATASET_NAME)
images_dir = os.path.join(data_dir, 'images')
labels_dir = os.path.join(data_dir, 'labels')

train_dataset = SedimentsDataset(images_dir, labels_dir,
                                 target_class=TARGET_CLASS, transform=transform, mode='train')
val_dataset = SedimentsDataset(images_dir, labels_dir, target_class=TARGET_CLASS, mode='val')
test_dataset = SedimentsDataset(images_dir, labels_dir, target_class=TARGET_CLASS, mode='test')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Verify dataloader sizes
print(f'Training set size:   {len(train_loader.dataset)}')
print(f'Validation set size: {len(val_loader.dataset)}')
print(f'Testing set size:    {len(test_loader.dataset)}')

# It is a good practice to check datasets do not intersects with each other
assert set(test_dataset.ids).isdisjoint(set(train_dataset.ids))
assert set(test_dataset.ids).isdisjoint(set(val_dataset.ids))
assert set(train_dataset.ids).isdisjoint(set(val_dataset.ids))

for sample in [train_dataset[0], val_dataset[0], test_dataset[0]) :
  plt.subplot(121)
  plt.imshow(sample['image'].transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
  plt.subplot(122)
  plt.imshow(sample['mask'].squeeze())  # for visualization we have to remove 3rd dimension of mask
  plt.show()

best_model_checkpoint = ModelCheckpoint(
    monitor='valid_dataset_iou',  # Track validation loss
    mode='max',    # Save model with the lowest val_loss
    save_top_k=1,  # Keep only the best model
    filename='best_model-{epoch:02d}-{valid_dataset_iou:.2f}'
)

# Last epoch checkpoint
last_epoch_checkpoint = ModelCheckpoint()

if (DATE_DIR_NAME is not None):
  date_dir_name = DATE_DIR_NAME
  if TRAIN:
    ckpt_path = get_most_recent_model(os.path.join(COLAB_PATH, 'Results', date_dir_name))
    model = SedimentsModel(ARCH, ENCODER, encoder_weights = 'imagenet', in_channels=3, out_classes=1)
    trainer = pl.Trainer(callbacks=[best_model_checkpoint, last_epoch_checkpoint], max_epochs=MAX_EPOCHS,
                         default_root_dir=os.path.join(COLAB_PATH, 'Results', date_dir_name))
    trainer.fit(model, ckpt_path = ckpt_path, train_dataloaders=train_loader, val_dataloaders=val_loader,)
  else:
    ckpt_path = get_best_model(os.path.join(COLAB_PATH, 'Results', date_dir_name))
    model = SedimentsModel.load_from_checkpoint(ckpt_path)
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS,
                         default_root_dir=os.path.join(COLAB_PATH, 'Results', date_dir_name))
else:
  model = SedimentsModel(ARCH, ENCODER, encoder_weights = 'imagenet', in_channels=3, out_classes=1)
  date_dir_name = f'{USER}_{TARGET_CLASS}_{ARCH}_{ENCODER}_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  trainer = pl.Trainer(callbacks=[best_model_checkpoint, last_epoch_checkpoint], max_epochs=MAX_EPOCHS,
                       default_root_dir=os.path.join(COLAB_PATH, 'Results', date_dir_name))
  trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# run validation dataset
valid_metrics = trainer.validate(model, dataloaders=val_loader, verbose=False)
pprint(valid_metrics)

# run test dataset
test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
pprint(test_metrics)

# create results file
results_file_path = os.path.join(COLAB_PATH, 'Results', date_dir_name, 'IoU.txt')
with open(results_file_path, 'w') as file:
    file.write(f'Class: {TARGET_CLASS}\n')
    file.write(f'Architecture: {ARCH}\n')
    file.write(f'Encoder: {ENCODER}\n')
    for key, value in valid_metrics[0].items():
        file.write(f'{key}: {value}\n')
    for key, value in test_metrics[0].items():
        file.write(f'{key}: {value}\n')

# results visualization and saving images
result_images_paths = os.path.join(COLAB_PATH, 'Results', date_dir_name, 'Test images')
if not os.path.exists(result_images_paths):
  os.makedirs(result_images_paths)

batch = next(iter(test_loader))
with torch.no_grad():
    model.eval()
    logits = model(batch['image'])
pr_masks = logits.sigmoid()

cont = 1
for image, gt_mask, pr_mask in zip(batch['image'], batch['mask'], pr_masks):
    plt.figure(figsize=(10, 5))

    plt.subplot(131)
    plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
    plt.title('Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(gt_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
    plt.title('Ground truth')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(pr_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
    plt.title('Prediction')
    plt.axis('off')
    plt.savefig(os.path.join(result_images_paths, f'images_{cont}.png'))

    plt.show()
    cont += 1
