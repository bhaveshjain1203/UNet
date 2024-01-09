import torch
import torchvision
from load_dataset import ImageDataset
from torch.utils.data import DataLoader
from model import Unet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
import albumentations as A
from albumentations.pytorch import ToTensorV2


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    # put model to evaluation mode
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)

        with torch.no_grad(): # disable gradients
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    # switch to training mode
    model.train()

    
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
IMAGE_HEIGHT = 256  
IMAGE_WIDTH = 256 

PIN_MEMORY = False
LOAD_MODEL = False
NUM_WORKERS = 0

TRAIN_IMG_DIR = "train_images_with_masks/images/"
TRAIN_MASK_DIR = "train_images_with_masks/masks/"
VAL_IMG_DIR = "val_images_with_masks/images/"
VAL_MASK_DIR = "val_images_with_masks/masks/"
model = Unet(inChannels=3, outChannels=1).to(DEVICE)
train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # A.Rotate(limit=35, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

save_predictions_as_imgs(
            train_loader, model, folder="saved_images/", device=DEVICE
        )