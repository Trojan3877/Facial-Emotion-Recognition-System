import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from src.config.settings import settings


EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}


# -------------------------------------------------------------
# OPTION A — Lightweight CNN (Fast, portable, simple to deploy)
# -------------------------------------------------------------

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (1,48,48) → (32,24,24)
        x = self.pool(F.relu(self.conv2(x)))  # → (64,12,12)
        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -------------------------------------------------------------
# OPTION B — ResNet18 Backbone (High accuracy)
# -------------------------------------------------------------

class ResNetEmotion(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNetEmotion, self).__init__()

        # Load ResNet-18 from torchvision
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify input to accept grayscale images (1 channel)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace output classifier
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# -------------------------------------------------------------
# MODEL WRAPPER FOR LOADING + INFERENCE
# -------------------------------------------------------------

class EmotionModel:
    """
    Wrapper that:
    - Loads the trained PyTorch FER model
    - Performs forward inference
    - Maps output logits → predicted label
    """

    def __init__(self, model_path=settings.MODEL_PATH, use_resnet=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_resnet = use_resnet
        
        # Select architecture
        if self.use_resnet:
            self.model = ResNetEmotion()
        else:
            self.model = EmotionCNN()

        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, face_tensor):
        """
        face_tensor shape: (1,1,48,48) or (1,1,224,224)
        Returns: {label: str, confidence: float}
        """
        with torch.no_grad():
            face_tensor = torch.tensor(face_tensor, dtype=torch.float32).to(self.device)
            outputs = self.model(face_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()

        return {
            "emotion": EMOTION_LABELS[predicted_idx],
            "confidence": round(float(confidence), 4)
        }
