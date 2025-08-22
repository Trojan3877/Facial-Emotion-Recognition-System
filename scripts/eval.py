# scripts/eval.py
import argparse, os, itertools
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import tensorflow as tf

DEFAULT_CLASSES = ["angry","disgust","fear","happy","sad","surprise","neutral"]

def load_image_paths_labels(root_dir, classes):
    """Load file paths and integer labels from test/ subdir."""
    img_paths, labels = [], []
    for idx, cls in enumerate(classes):
        folder = Path(root_dir) / "test" / cls
        if not folder.exists():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        for p in folder.glob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                img_paths.append(p.as_posix())
                labels.append(idx)
    return img_paths, np.array(labels, dtype=np.int32)

def preprocess(img: Image.Image, img_size: int, grayscale: bool):
    if grayscale:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    img = img.resize((img_size, img_size))
    x = np.asarray(img, dtype=np.float32) / 255.0
    if grayscale:
        x = np.expand_dims(x, axis=-1)
    return x

def plot_confusion_matrix(cm, classes, normalize=True, out="assets/confusion_matrix.png"):
    os.makedirs(Path(out).parent, exist_ok=True)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
    plt.figure(figsize=(7,6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=8)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()

def main(args):
    classes = args.classes.split(",") if args.classes else DEFAULT_CLASSES
    img_paths, y_true = load_image_paths_labels(args.data, classes)

    # Optionally limit for CI speed
    if args.limit and args.limit > 0:
        img_paths = img_paths[:args.limit]
        y_true = y_true[:args.limit]

    model = tf.keras.models.load_model(args.weights)

    X = []
    for p in img_paths:
        img = Image.open(p)
        x = preprocess(img, args.img_size, args.grayscale == 1)
        X.append(x)
    X = np.stack(X, axis=0)

    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    print(f"ACCURACY: {acc:.4f}")
    print(f"F1_MACRO: {f1_macro:.4f}")
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    plot_confusion_matrix(cm, classes, normalize=True, out="assets/confusion_matrix.png")
    print("Saved confusion matrix to assets/confusion_matrix.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/fer2013", help="dataset root with test/ subdir")
    p.add_argument("--weights", type=str, default="model/emotion_model.h5", help="Keras model path")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--grayscale", type=int, default=0, help="1 = grayscale pipeline")
    p.add_argument("--classes", type=str, default=",".join(DEFAULT_CLASSES))
    p.add_argument("--limit", type=int, default=0, help="limit images (useful for CI)")
    args = p.parse_args()
    main(args)
