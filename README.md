# Facial Emotion Recognition System

![CI](https://img.shields.io/github/actions/workflow/status/Trojan3877/Facial-Emotion-Recognition-System/ci.yml?branch=main)
![License](https://img.shields.io/github/license/Trojan3877/Facial-Emotion-Recognition-System)
![GitHub stars](https://img.shields.io/github/stars/Trojan3877/Facial-Emotion-Recognition-System?style=social)
![GitHub forks](https://img.shields.io/github/forks/Trojan3877/Facial-Emotion-Recognition-System?style=social)
![GitHub issues](https://img.shields.io/github/issues/Trojan3877/Facial-Emotion-Recognition-System)
![GitHub top language](https://img.shields.io/github/languages/top/Trojan3877/Facial-Emotion-Recognition-System)

>Your one-line project description here...


# Facial-Emotion-Recognition-System
Facial Emotion Recognition System Model
# Facial Emotion Recognition System 🎭

![image](https://github.com/user-attachments/assets/3e376a84-c942-468c-9652-30d1d947ac04)


## 📌 Project Overview
This project builds a deep learning-based Facial Emotion Recognition System using the FER2013 dataset. It leverages a Convolutional Neural Network (CNN) to classify facial expressions into seven emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

Designed as a GitHub capstone project for showcasing AI/ML skills, this system includes modular, commented Python code, visualizations, and quantifiable evaluation metrics—all without requiring local deployment.

---

## 🎯 Objective
To develop a robust, image-based emotion recognition model using supervised learning, enabling real-time or batch classification of facial expressions for use in smart environments, user behavior analysis, and AI-human interaction.

---

## 📊 Dataset
- **Name:** FER2013 (Facial Expression Recognition 2013)
- **Source:** [Kaggle / GitHub Mirror](https://github.com/gitshanks/fer2013)
- **Samples:** 35,887 grayscale images (48x48 pixels)
- **Labels:** 7 Emotion Classes  
  `['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']`

---

## 🧰 Tech Stack
- **Language:** Python
- **Libraries:** TensorFlow, Keras, Pandas, NumPy, OpenCV, Scikit-learn, Matplotlib, Seaborn
- **Tools:** Git, GitHub, Jupyter Notebook, YAML
- **Architecture:** CNN with dropout, softmax classifier, and early stopping

---

## 📂 Project Structure

facial-emotion-recognition/ │ ├── config/ │ └── config.yaml ├── data/ │ ├── raw/fer2013.csv │ └── processed/ ├── src/ │ ├── data_loader.py │ ├── preprocess.py │ ├── train.py │ ├── evaluate.py │ └── visualize.py ├── models/ │ └── emotion_cnn.h5 ├── notebooks/ │ └── exploration.ipynb ├── README.md └── requirements.txt

---

## 📈 Evaluation Metrics
- **Accuracy**
- **Precision, Recall, F1-Score**
- **Confusion Matrix**
- **Training vs Validation Loss/Accuracy Curves**

---

## 🖼️ Visuals
- Class distribution plots  
- Sample emotion-labeled images  
- Confusion matrix heatmap  
- Model training performance graphs

---

## 🚫 Limitations
- Not deployed due to local GPU limitations
- FER2013 images are low-res and may not generalize well in production

---

## 💡 Future Enhancements
- Integrate with webcam for real-time emotion detection
- Upgrade to ResNet or MobileNet for performance boost
- Deployment via Flask + Docker or Streamlit when hardware allows

---

## 👨‍💻 Author
**Corey Leath**  
## 🧪 Run the Notebook

To explore the data or preview facial samples, open the Jupyter notebook:

```bash
jupyter notebook notebooks/exploration.ipynb

GitHub: [github.com/Trojan3877](https://github.com/Trojan3877)  
LinkedIn: [linkedin.com/in/corey-leath](https://www.linkedin.com/in/corey-leath)

---

## 📜 License
This project is open-source under the MIT License.
