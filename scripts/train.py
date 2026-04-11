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
BATCH_SIZE = 2
EPOCHS = 10
LR = 1e-3

DEVICE = "cpu"  # force CPU for stability

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
            f"{IMAGES_DIR}/{img_id}.png"
        ).convert("RGB")

        mask = Image.open(
            f"{MASKS_DIR}/{img_id}.png"
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
# Lightweight U-Net
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
# Training
# -------------------------
def main():
    train_ds = SegmentationDataset(f"{SPLITS_DIR}/train.txt")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = MiniUNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0

        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {loss_sum:.4f}"
        )

    torch.save(model.state_dict(), "model.pt")
    print("✅ Lightweight model saved as model.pt")

if __name__ == "__main__":
    main()