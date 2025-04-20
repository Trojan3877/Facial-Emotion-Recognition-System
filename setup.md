# 🛠️ Setup Instructions for Facial Emotion Recognition System

These instructions help you set up and run the full project environment locally using Python and Jupyter Notebook.

---

## ✅ 1. Clone the Repository

```bash
git clone https://github.com/Trojan3877/facial-emotion-recognition.git
cd facial-emotion-recognition

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

python src/preprocess.py
python src/train.py

python src/evaluate.py

python src/visualize.py

jupyter notebook notebooks/exploration.ipynb

