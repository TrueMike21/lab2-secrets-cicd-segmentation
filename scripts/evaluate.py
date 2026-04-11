import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image

# -------------------------
# Configuration
# -------------------------
IMAGES_DIR = "data/raw/aerial_images"
MASKS_DIR = "data/processed/masks"
SPLITS_DIR = "data/processed/splits"

IMAGE_SIZE = 256
BATCH_SIZE = 1
DEVICE = "cpu"

# -------------------------
# Dataset
# -------------------------
class SegmentationDataset(Dataset):
    def __init__(self, split_file):
        with open(split_file) as f:
            self.ids = [line.strip() for line in f]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        image = Image.open(
            os.path.join(IMAGES_DIR, f"{img_id}.png")
        ).convert("RGB")

        mask = Image.open(
            os.path.join(MASKS_DIR, f"{img_id}.png")
        ).convert("L")

        image = F.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        mask = F.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))

        image = torch.from_numpy(
            np.array(image).transpose(2, 0, 1) / 255.0
        ).float()

        mask = torch.from_numpy(
            (np.array(mask) > 0).astype(np.float32)
        ).unsqueeze(0)

        return image, mask

# -------------------------
# Lightweight Model (same as train.py)
# -------------------------
class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )
        self.up = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.enc2(x2)
        x4 = self.up(x3)
        return self.out(x4)

# -------------------------
# Metrics
# -------------------------
def compute_iou(pred, target):
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    if union == 0:
        return 1.0
    return intersection / union

def compute_dice(pred, target):
    pred_sum = pred.sum()
    target_sum = target.sum()

    # both empty → perfect match
    if pred_sum == 0 and target_sum == 0:
        return 1.0

    intersection = (pred & target).sum()
    return (2 * intersection) / (pred_sum + target_sum)
# -------------------------
# Evaluation
# -------------------------
def main():
    dataset = SegmentationDataset(os.path.join(SPLITS_DIR, "test.txt"))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # ✅ Recreate model and load weights correctly
    model = MiniUNet().to(DEVICE)
    model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
    model.eval()

    iou_scores = []
    dice_scores = []

    with torch.no_grad():
        for image, mask in loader:
            image = image.to(DEVICE)
            mask = mask.to(DEVICE)

            logits = model(image)
            pred = (torch.sigmoid(logits) > 0.5).int()
            target = mask.int()

            
            iou_scores.append(compute_iou(pred, target))
            dice_scores.append(compute_dice(pred, target))


    print("✅ Evaluation complete")
    print(f"Mean IoU  : {sum(iou_scores) / len(iou_scores):.4f}")
    print(f"Mean Dice : {sum(dice_scores) / len(dice_scores):.4f}")

if __name__ == "__main__":
    main()