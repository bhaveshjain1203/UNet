import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        # load image
        # image is np array bcz augmentation lib is used ,which when using PIL needed to be np array ??? 
        image = np.array(Image.open(img_path).convert("RGB"))

        # load mask -> L = grayscale
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        # sigmoid is used as last activation
        mask[mask == 255.0] = 1.0

        # Data augmentation - ???
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

