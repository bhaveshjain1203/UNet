import torch
import torchvision
from load_dataset import ImageDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=0,
    pin_memory=False,
):
    train_ds = ImageDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = ImageDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    # Number of correctly predicted pixels
    num_correct = 0

    # Total number of pixels
    num_pixels = 0

    dice_score = 0
    
    ''' putting model in evaluation mode,
        This is important because some layers, like dropout or batch normalization, 
        behave differently during training and evaluation
    '''
    model.eval() 

    # disable the gradients,done to prevent uneccesary calculations during inference
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            
            # pass the input to model and get the predictions,torch.sigmoid is used to squeeze the output between 0 and 1
            preds = torch.sigmoid(model(x))
            
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()

            # torch.numel - returns total number of elements in preds
            num_pixels += torch.numel(preds)

            # check - intersection over union ??
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )

    # print avg dice score across all batches
    print(f"Dice score: {dice_score/len(loader)}")

    # set model back to training mode
    model.train()

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

