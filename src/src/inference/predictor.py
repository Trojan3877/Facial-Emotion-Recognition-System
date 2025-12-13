import cv2
import numpy as np
from src.detection.detect_faces import FaceDetector
from src.modeling.model import EmotionModel
from src.config.settings import settings


class EmotionPredictor:
    """
    Inference engine that:
    - Detects faces
    - Preprocesses them
    - Runs emotion classification
    - Returns structured predictions for LLM + RAG integration
    """

    def __init__(self, use_resnet=False):
        self.detector = FaceDetector()
        self.model = EmotionModel(
            model_path=settings.MODEL_PATH,
            use_resnet=use_resnet
        )

    def load_image(self, image_path):
        """
        Loads image from file path into BGR numpy array.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at: {image_path}")
        return image

    def predict_from_array(self, image_array):
        """
        Accepts a numpy array image (BGR).
        Returns prediction dictionary.
        """
        faces = self.detector.extract_and_preprocess(image_array)

        if len(faces) == 0:
            return {
                "error": "No faces detected in image.",
                "predictions": []
            }

        results = []
        for idx, face_tensor in enumerate(faces):
            prediction = self.model.predict(face_tensor)
            results.append({
                "face_id": idx + 1,
                "emotion": prediction["emotion"],
                "confidence": prediction["confidence"]
            })

        return {"predictions": results}

    def predict_from_path(self, image_path):
        """
        Full pipeline for image file input:
        1. Load image
        2. Detect + preprocess
        3. Predict emotion
        """
        image = self.load_image(image_path)
        return self.predict_from_array(image)

    def predict_and_format_for_llm(self, image_array):
        """
        Provides enhanced output specially formatted for LLM explanation.

        Example output:
        {
            "emotions": ["Happy", "Neutral"],
            "confidences": [0.92, 0.77],
            "summary": "2 faces detected. Highest emotion = Happy."
        }
        """
        result = self.predict_from_array(image_array)

        if "error" in result:
            return result

        emotions = [p["emotion"] for p in result["predictions"]]
        confidences = [p["confidence"] for p in result["predictions"]]

        summary = (
            f"{len(emotions)} face(s) detected. "
            f"Primary emotion: {emotions[0]} "
            f"(confidence: {confidences[0]})."
        )

        return {
            "emotions": emotions,
            "confidences": confidences,
            "summary": summary
        }
