# scripts/prepare_fer2013.py
import csv, os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# FER2013 label map
IDX2NAME = {
    "0": "angry",
    "1": "disgust",
    "2": "fear",
    "3": "happy",
    "4": "sad",
    "5": "surprise",
    "6": "neutral",
}

def ensure(p: Path): 
    p.mkdir(parents=True, exist_ok=True)

def row_to_img(pixels: str, size=(48,48)):
    vals = np.array([int(x) for x in pixels.split()], dtype=np.uint8)
    img = vals.reshape(size)
    return Image.fromarray(img, mode="L")

def main(csv_path="fer2013.csv", out_root="data/fer2013"):
    out = Path(out_root)
    for split in ["train","val","test"]:
        for cls in IDX2NAME.values():
            ensure(out / split / cls)

    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        counters = {}
        for row in tqdm(r, desc="Exporting FER2013"):
            emotion = row["emotion"]
            usage = row["Usage"].lower()  # "Training", "PublicTest", "PrivateTest"
            pixels = row[" pixels"].strip() if " pixels" in row else row["pixels"].strip()

            if usage.startswith("train"):
                split = "train"
            elif usage.startswith("public"):
                split = "val"
            else:
                split = "test"

            cls = IDX2NAME[emotion]
            key = (split, cls)
            counters[key] = counters.get(key, 0) + 1
            idx = counters[key]

            img = row_to_img(pixels)
            # save as PNG
            img.save(out / split / cls / f"{cls}_{idx:05d}.png")

    print(f"Done. Folderized dataset at: {out.resolve()}")

if __name__ == "__main__":
    main()
