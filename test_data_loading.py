import os

def test_dataset_exists():
    assert os.path.exists("data"), "❌ data/ folder not found"

def test_sample_image_shape():
    import cv2
    img = cv2.imread("data/sample.jpg")
    assert img is not None, "❌ sample image not loading"
    assert len(img.shape) == 3, "❌ image must have 3 channels"
