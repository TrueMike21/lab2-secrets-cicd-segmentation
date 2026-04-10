import os
import json
import cv2
import numpy as np
from pathlib import Path

# ----------------------------
# Configuration
# ----------------------------
IMAGES_DIR = "data/raw/aerial_images"
ANNOTATIONS_DIR = "data/raw/annotations"   # GeoJSON files
OUTPUT_MASKS_DIR = "data/processed/masks"

IMAGE_EXT = ".jpg"  # or .png


def create_mask(image_shape, polygons):
    """
    Create a binary mask from polygon coordinates.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for polygon in polygons:
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    return mask


def load_polygons_from_geojson(geojson_path):
    """
    Extract polygon coordinates from a GeoJSON file.
    """
    with open(geojson_path, "r") as f:
        geojson = json.load(f)

    polygons = []

    for feature in geojson["features"]:
        geometry = feature["geometry"]

        if geometry["type"] == "Polygon":
            polygons.extend(geometry["coordinates"])
        elif geometry["type"] == "MultiPolygon":
            for poly in geometry["coordinates"]:
                polygons.extend(poly)

    return polygons


def main():
    Path(OUTPUT_MASKS_DIR).mkdir(parents=True, exist_ok=True)

    for image_file in os.listdir(IMAGES_DIR):
        if not image_file.endswith(IMAGE_EXT):
            continue

        image_path = os.path.join(IMAGES_DIR, image_file)
        annotation_path = os.path.join(
            ANNOTATIONS_DIR, image_file.replace(IMAGE_EXT, ".geojson")
        )

        if not os.path.exists(annotation_path):
            print(f"Annotation missing for {image_file}")
            continue

        image = cv2.imread(image_path)
        polygons = load_polygons_from_geojson(annotation_path)
        mask = create_mask(image.shape, polygons)

        mask_path = os.path.join(
            OUTPUT_MASKS_DIR, image_file.replace(IMAGE_EXT, ".png")
        )

        cv2.imwrite(mask_path, mask)
        print(f"Saved mask: {mask_path}")


if __name__ == "__main__":
    main()