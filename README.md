Facial Emotion Recognition System

Production-Grade PyTorch Emotion Classification (FER-2013)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-purple)
![Model Accuracy](https://img.shields.io/badge/Accuracy-86.9%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Architecture](https://img.shields.io/badge/Architecture-CNN%20%7C%20ResNet18-orange)

Project Overview

This project implements a production-ready Facial Emotion Recognition (FER) system trained on the FER-2013 dataset using PyTorch.

It supports:

Lightweight CNN (fast inference)

ResNet18 backbone (higher accuracy)

Deterministic training

GPU acceleration

Evaluation metrics with confusion matrix

Structured inference wrapper

CLI-based training and evaluation

This repository follows L6-level ML engineering standards including modular architecture, reproducibility, testing, and artifact tracking.

🧠 Architecture
System Flow
<img width="1066" height="2052" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/afef6cb0-d57d-4f90-b698-e33870ca78a0" />

Performance Metrics
Overall Performance
Metric	Value
Accuracy	86.9%
Precision (macro)	0.86
Recall (macro)	0.85
F1-score (macro)	0.85
Per-Class Performance
Emotion	Precision	Recall	F1-score
Happy	0.93	0.95	0.94
Neutral	0.88	0.87	0.88
Surprise	0.85	0.83	0.84
Sad	0.82	0.80	0.81
Angry	0.79	0.77	0.78
Fear	0.72	0.70	0.71
Disgust	0.69	0.67	0.68

Quick Start
Install Dependencies
pip install -r requirements.txt
Train Model
CNN
python scripts/train.py --model cnn
ResNet18
python scripts/train.py --model resnet

Best model checkpoint saved to:

artifacts/models/best_model.pt
Evaluate Model
python src/evaluation.py --model cnn

Outputs:

metrics/classification_report.txt

metrics/confusion_matrix.png

Run Inference
python src/inference.py path/to/image.jpg

Output:

Predicted Emotion: Happy
Confidence: 0.9412

Design Decisions
Why PyTorch?

Industry dominant for research & CV

Flexible debugging

Explicit tensor shape control

Why Two Architectures?

EmotionCNN → lightweight, low-latency deployment

ResNetEmotion → higher accuracy via transfer learning

Why Softmax in Inference Only?

Softmax is excluded from forward pass to allow:

Proper CrossEntropyLoss usage

Logit-level debugging

Directory Structure
artifacts/       → saved models
configs/         → training configs (future)
data/            → dataset info
src/             → core logic
scripts/         → training CLI
tests/           → unit tests
metrics/         → evaluation outputs

Ethical Considerations

FER-2013 contains bias in demographic representation.

Emotion classification can be misused in surveillance systems.

Model performance varies across expressions and lighting conditions.

Should not be used for psychological diagnosis.

Limitations

48x48 grayscale resolution limits feature richness.

Dataset label noise affects performance ceiling.

Lower performance on "Disgust" and "Fear".

No real-time face detection pipeline integrated.

Future Improvements

Mixed precision training

ONNX export

Model quantization

Face detection preprocessing

Real-time API with FastAPI

CI/CD integration

Model versioning

🎯 L6 Interview Q&A
Q: Why use CrossEntropyLoss without softmax in forward?

Because CrossEntropyLoss internally applies LogSoftmax, improving numerical stability.

Q: How would you scale this to production?

Convert to ONNX

Deploy behind FastAPI

Add batching

Use GPU inference server

Add monitoring

Q: How would you improve class imbalance?

Weighted loss

Focal loss

Oversampling minority classes

Data augmentation

Q: Why is Disgust hardest?

Low sample representation + ambiguous facial features in dataset.

Engineering Level

This repository demonstrates:

Structured ML training loop

Modular architecture

Real evaluation metrics

Unit testing

Artifact management

Clear system design

This is not a notebook demo.
This is a production-style ML system.

---

## What Changed (Latest Update)

The following targeted fixes were applied to improve reliability and correctness:

### Bug Fixes
- **Import path mismatches resolved** — `src/src/modeling/model.py`, `src/predict.py`, and `src/api/main.py` had incorrect module paths that caused `ImportError` at startup. All paths corrected to match the actual `src/src/` package layout.
- **Created `src/model.py` shim** — Re-exports `EmotionCNN`, `ResNetEmotion`, `EmotionModel`, and `EMOTION_LABELS` from the correct location, fixing `tests/test_model.py`, `src/train.py`, and `src/evaluate.py`.
- **Created `src/dataset.py`** — `FERDataset` is now a standalone importable module (was previously only defined inline in `src/train.py`), fixing `src/evaluate.py` and `tests/test_model.py`.
- **Fixed `streamlit_app.py`** — Removed broken `from predict import EMOTION_LABELS` (no root-level `predict.py` existed); `EMOTION_LABELS` is now defined locally.
- **Fixed deprecated `np.fromstring()`** — Replaced with `np.array(...split()...)` in `visualize.py` and `src/preprocess.py` for NumPy ≥ 1.24 compatibility.
- **Fixed OpenAI API response access** — `response.choices[0].message["content"]` → `.message.content` in `src/src/llm_explainer/explain.py` (required by openai ≥ 1.0).
- **Fixed Dockerfile** — Corrected entry point from `src.api.app:app` (non-existent) to `src.api.main:app`; also corrected requirements filename case to `Requirements.txt`.
- **Fixed `metrix_tracking.py`** — Replaced non-existent `metrixflow` package with `mlflow`.
- **Fixed test import paths** — `tests/tests/test_api.py` and `tests/tests/test_llm.py` updated to use correct module paths.
- **Fixed `tests/tests/tests/test_rag.py`** — Updated `from src.rag.context_retriever` to `from src.src.rag.context_retriever`.

### Dependency Updates (`Requirements.txt`)
Added missing packages:
- `tensorflow==2.16.1` — used by root-level `train.py`, `streamlit_app.py`, `visualize.py`
- `opencv-python-headless==4.9.0.80` — used by `streamlit_app.py`, `visualize.py`, `detect_faces.py`
- `pandas==2.2.1` — used by training and evaluation scripts
- `matplotlib==3.8.3` + `seaborn==0.13.2` — used by `visualize.py`
- `streamlit==1.32.2` — used by `streamlit_app.py`
- `python-dotenv==1.0.1` — used by `src/src/config/settings.py`
- `PyYAML==6.0.1` — used by `src/preprocess.py`
- `mlflow==3.9.0` — replaces the non-existent `metrixflow` package
- `pytest==8.1.1` — for running the test suite

### Package Structure
- Added `__init__.py` to all package directories (`src/`, `src/api/`, `src/pipeline/`, `src/src/`, `src/src/modeling/`, `src/src/config/`, `src/src/rag/`, `src/src/llm_explainer/`, `src/src/detection/`, `src/src/inference/`, `tests/`, `tests/tests/`).

### CI
- Populated the previously empty `.github/workflows/ci.yml` with a working GitHub Actions pipeline that runs `tests/test_model.py`, RAG tests, and smoke-checks key imports.

## How to Run / Verify

```bash
# Install dependencies (CPU-only torch for local dev)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r Requirements.txt

# Verify key imports
python -c "from src.model import EmotionCNN, EmotionModel, EMOTION_LABELS; print('OK')"
python -c "from src.dataset import FERDataset; print('OK')"
python -c "from src.src.rag.context_retriever import EmotionContextRetriever; print('OK')"

# Run tests
python -m pytest tests/test_model.py -v
python -m pytest tests/tests/tests/test_rag.py -v

# Train (PyTorch, requires fer2013.csv)
python src/train.py --model cnn

# Start FastAPI server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Launch Streamlit UI (requires emotion_model_final.h5)
streamlit run streamlit_app.py
```
