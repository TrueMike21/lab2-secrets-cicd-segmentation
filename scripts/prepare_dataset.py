import os
import random
from pathlib import Path

IMAGES_DIR = "data/raw/aerial_images"
SPLITS_DIR = "data/processed/splits"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def main():
    Path(SPLITS_DIR).mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        f.replace(".png", "")
        for f in os.listdir(IMAGES_DIR)
        if f.endswith(".png")
    ])

    random.shuffle(image_files)

    n = len(image_files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = image_files[:n_train]
    val = image_files[n_train:n_train + n_val]
    test = image_files[n_train + n_val:]

    def save_split(name, data):
        with open(os.path.join(SPLITS_DIR, f"{name}.txt"), "w") as f:
            for item in data:
                f.write(item + "\n")

    save_split("train", train)
    save_split("val", val)
    save_split("test", test)

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

if __name__ == "__main__":
    main()