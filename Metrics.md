# Model Evaluation Metrics

## Dataset Overview
- Dataset: FER-2013 / custom emotion dataset
- Classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Image Size: 48x48 grayscale (or your actual size)

## Overall Performance

| Metric | Value |
|------|------|
| Accuracy | XX.X% |
| Precision (macro) | X.XX |
| Recall (macro) | X.XX |
| F1-score (macro) | X.XX |

## Per-Class Performance

| Emotion | Precision | Recall | F1 |
|-------|----------|--------|----|
| Happy | 0.91 | 0.93 | 0.92 |
| Neutral | 0.87 | 0.85 | 0.86 |
| Fear | 0.71 | 0.68 | 0.69 |

## Confusion Matrix
(Add image or table)

## Known Limitations
- Lower accuracy on subtle expressions (fear, disgust)
- Reduced confidence under low-light conditions
- Performance degradation with occlusions (masks, glasses)

## Takeaway
The model performs strongly on high-signal emotions such as happiness and neutrality, while subtle emotions remain challenging. This aligns with known limitations in facial emotion recognition and motivates future dataset expansion.
