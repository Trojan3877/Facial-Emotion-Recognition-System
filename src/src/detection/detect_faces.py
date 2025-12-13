import cv2
import numpy as np

class FaceDetector:
    """
    Face detection and preprocessing module.
    Uses OpenCV Haar Cascades for lightweight detection.
    """

    def __init__(self, face_cascade_path="haarcascade_frontalface_default.xml"):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)

    def detect_faces(self, image):
        """
        Detects faces in an image.
        Returns list of cropped face regions.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        cropped_faces = []
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            cropped_faces.append(face)

        return cropped_faces

    def preprocess_face(self, face, img_size=(48, 48)):
        """
        Converts face image into grayscale and resizes it
        for CNN/ResNet FER models.
        """
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, img_size)
        face_normalized = face_resized.astype("float32") / 255.0
        face_expanded = np.expand_dims(face_normalized, axis=0)
        face_expanded = np.expand_dims(face_expanded, axis=0)  # final shape: (1,1,48,48)

        return face_expanded

    def extract_and_preprocess(self, image):
        """
        Pipeline function that:
        1. Detects faces
        2. Preprocesses them
        Returns list of preprocessed face tensors.
        """
        faces = self.detect_faces(image)
        preprocessed = [self.preprocess_face(face) for face in faces]
        return preprocessed
