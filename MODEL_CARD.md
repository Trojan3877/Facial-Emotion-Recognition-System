# 🧠 **MODEL_CARD.md — Facial Emotion Recognition System**

### **Author:** Corey Leath (Trojan3877)

### **Version:** v1.0

### **Last Updated:** 2025-11-15

---

# 📌 **1. Model Overview**

The **Facial Emotion Recognition (FER) Model** is a Convolutional Neural Network trained on the **FER-2013 dataset** to classify human facial expressions into seven emotion categories:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

This model is deployed using **FastAPI**, **Streamlit**, **Docker**, and **Kubernetes**, and tracked using **MLflow**. Prediction logs are stored using **Snowflake** for enterprise-grade MLOps.

---

# 🎯 **2. Intended Use**

### ✔ **Primary use case**

To provide real-time emotion recognition for:

* Education tools
* Interactive AI demos
* User behavior research
* Human–computer interaction (HCI)
* Wellness monitoring applications
* Social robotics
* UX or product analytics

### ⚠ **Not intended for:**

* Law enforcement
* Hiring decision tools
* Predicting personality
* Medical diagnosis
* Legal or governmental use
* Automated surveillance
* High-stakes decision-making

This model **should never** be used as the sole decision-maker for any sensitive task.

---

# 🧩 **3. Model Architecture**

* Architecture: **CNN (Convolutional Neural Network)**
* Framework: **TensorFlow/Keras**
* Input: 48×48 grayscale images
* Output: 7-class softmax probability distribution

Logged in MLflow:

✔ Model architecture
✔ Parameters
✔ Metrics
✔ Training loss curves
✔ Confusion matrix
✔ Artifacts (history.json)

---

# 📚 **4. Training Data**

**Dataset:** FER-2013 (public Kaggle dataset)

### Pros:

* Large dataset (~35,000 images)
* Multiple classes
* Standard benchmark in Computer Vision

### Cons:

* Dataset skews toward younger faces
* Overrepresentation of certain ethnic groups
* Limited lighting variations
* Some images mislabeled
* Grayscale only

---

# ⚠️ **5. Ethical Considerations & Known Biases**

Emotion recognition is an area with **well-documented ethical risks**, including:

### **1. Demographic Bias**

Studies show FER datasets often underrepresent:

* Darker skin tones
* Older adults
* Certain ethnic backgrounds

This can cause **misclassification**.

### **2. Cultural Expression Variability**

Emotions are expressed differently across:

* Culture
* Gender
* Region
* Social norms

This model does **not** account for cultural variation.

### **3. Misuse / Overuse**

FER models can be misused in:

* Workplace surveillance
* Student behavior scoring
* Criminal profiling
* Automated “emotion-based decisions”

This model **must not** be deployed in these contexts.

---

# 📊 **6. Performance & Evaluation**

### **Metrics Logged in MLflow**

* Accuracy
* Validation accuracy
* Precision, Recall, F1-score (optional)
* Confusion matrix
* Loss curves

### **Typical Performance (FER-2013 Benchmarks)**

(Note: These are reference-level until you run eval)

* Overall Accuracy: **0.64–0.72 typical range**
* Best accuracy tends to be on: **Happy, Surprise**
* Lowest accuracy typically on: **Disgust, Fear, Angry**

You may include exact numbers after running your model evaluation script.

---

# 🧪 **7. Limitations**

This model struggles with:

* Occluded faces (masks, glasses, hats)
* Low lighting
* Extreme angles
* Blurry images
* Cultural expression differences
* Mixed or ambiguous expressions
* Micro-expressions
* Faces not centered or cropped

The model expects **clean, centered, front-facing** 48×48 grayscale faces.

---

# 🧱 **8. Deployment Details**

### **Infrastructure**

* FastAPI for inference
* Streamlit for UI
* Docker containerization
* Kubernetes deployment with LoadBalancer
* MLflow experiment tracking
* Snowflake prediction logging
* GPU optional

### **APIs**

* `POST /predict`
* Accepts image upload
* Returns predicted emotion + confidence score

---

# 🔐 **9. Privacy & Security**

This system:

✔ Does **not** store user images locally
✔ Logs **only label + confidence** to Snowflake
✔ Uses Snowflake Secrets (env vars)
✔ Uses HTTPS when deployed
✔ Can anonymize all metadata

---

# 📝 **10. Future Improvements**

* Upgrade to a deeper CNN or ViT
* Bias reduction via dataset balancing
* Add adversarial robustness
* Add face detection preprocessor
* Add calibration for confidence scores
* Improve evaluation suite
* Add multilingual documentation
