import os
import numpy as np
import torch
from torch import nn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------
# Configuration
# -------------------------
IMAGES_DIR = "data/raw/aerial_images"
MASKS_DIR = "data/processed/masks"
SPLITS_DIR = "data/processed/splits"
OUTPUT_DIR = "reports/figures"

IMAGE_SIZE = 256
DEVICE = "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Model (same as training)
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
# Visualization
# -------------------------
def main():
    # Pick one test image
    with open(os.path.join(SPLITS_DIR, "test.txt")) as f:
        image_id = f.readline().strip()

    # Load image and GT mask
    image = Image.open(
        os.path.join(IMAGES_DIR, f"{image_id}.png")
    ).convert("L")   # force grayscale

    gt_mask = Image.open(
        os.path.join(MASKS_DIR, f"{image_id}.png")
    ).convert("L")

    image = F.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    gt_mask = F.resize(gt_mask, (IMAGE_SIZE, IMAGE_SIZE))

    image_np = np.array(image)
    gt_mask_np = np.array(gt_mask)

    # Load model
    model = MiniUNet().to(DEVICE)
    model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
    model.eval()

    # Predict mask
    image_tensor = torch.from_numpy(image_np / 255.0).float()
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    image_tensor = image_tensor.repeat(1, 3, 1, 1)  # fake RGB for model

    with torch.no_grad():
        pred = torch.sigmoid(model(image_tensor)) > 0.5
        pred_mask_np = pred.squeeze().cpu().numpy().astype(np.uint8) * 255

    # -------------------------
    # Plot (2 panels only)
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Aerial Image")
    axes[0].axis("off")

    
    axes[1].imshow(gt_mask_np, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")


    output_path = os.path.join(OUTPUT_DIR, "sample_prediction_bw.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"✅ Grayscale visualization saved to {output_path}")

if __name__ == "__main__":
    main()