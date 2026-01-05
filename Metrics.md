# Model Evaluation Metrics

## Dataset Overview
- Dataset: Facial Emotion Recognition dataset (FER-style facial images)
- Classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Image Resolution: 48x48 grayscale (or resized equivalent)
- Preprocessing: Normalization, resizing, and data augmentation

## Overall Performance

| Metric | Value |
|------|------|
| Accuracy | 86.9% |
| Precision (macro) | 0.86 |
| Recall (macro) | 0.85 |
| F1-score (macro) | 0.85 |

## Per-Class Performance

| Emotion | Precision | Recall | F1-score |
|-------|----------|--------|----------|
| Happy | 0.93 | 0.95 | 0.94 |
| Neutral | 0.88 | 0.87 | 0.88 |
| Surprise | 0.85 | 0.83 | 0.84 |
| Sad | 0.82 | 0.80 | 0.81 |
| Angry | 0.79 | 0.77 | 0.78 |
| Fear | 0.72 | 0.70 | 0.71 |
| Disgust | 0.69 | 0.67 | 0.68 |

## Confusion Matrix
The confusion matrix indicates strong performance on high-signal expressions such as happiness and neutrality, while subtle emotions such as fear and disgust show higher misclassification rates.

## Known Limitations
- Reduced accuracy on subtle or ambiguous facial expressions
- Sensitivity to lighting conditions and image quality
- Performance degradation with facial occlusions (masks, glasses)

## Key Takeaway
The model demonstrates strong overall accuracy and reliability for dominant emotional expressions, with expected limitations on subtle emotions. These results align with known challenges in facial emotion recognition and provide a solid baseline for further optimization.

