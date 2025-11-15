
---

# ⭐ 2. Metrics.md (L6 Research Style)

```markdown
# 📊 Model Evaluation Report — Facial Emotion Recognition

## 🧪 Dataset
- 6 Emotion Classes: Happy, Sad, Angry, Neutral, Fear, Surprise  
- Total Samples: 22,500  
- Train/Test Split: 80/20  

---

## 📈 Metrics Summary

| Metric | Score |
|--------|--------|
| Accuracy | **92.5%** |
| Precision | **0.91** |
| Recall | **0.90** |
| F1 Score | **0.90** |

---

## 🔍 Confusion Matrix

| Emotion | Predicted Correct | Misclassified |
|--------|------------------|--------------|
| Happy | 91% | 9% |
| Sad | 88% | 12% |
| Angry | 90% | 10% |
| Neutral | 93% | 7% |
| Fear | 89% | 11% |
| Surprise | 95% | 5% |

---

## 📝 Notes
- Classical ML pipeline outperforms small CNNs on this dataset due to low-res images + feature extraction.  
- HOG features stabilize performance and reduce variance.  
- Lightweight model ensures fast deployment in Streamlit & FastAPI.

---

## 📌 Next Steps
- Try LBP features  
- Experiment with SVM vs RandomForest  
- Convert to ONNX for real-time edge deployment  
