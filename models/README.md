# Model Weights Directory

This folder stores the trained CNN model used for emotion prediction.

## 📌 File Required
Place your PyTorch model file here:


This file **should NOT be committed to GitHub** because:
- It may be large (50–200MB)
- It may contain training-specific metadata
- It increases cloning time for other users
- GitHub is not ideal for storing heavy ML artifacts

## 🚀 How to Use

1. Train your CNN model or download your pre-trained weights.
2. Save the file as:


3. Ensure the `MODEL_PATH` in `.env` matches:


4. When running Docker or FastAPI, the inference pipeline will automatically load the model.

## 🔒 Why This Folder Exists
This is a standard best practice in ML Engineering:

- Store code that references models  
- Store documentation for models  
- Do **not** store the actual weights in GitHub  
- Keep your repo lightweight and professional  

## 📝 Notes for Recruiters & Reviewers
This project intentionally excludes the model weights to keep the repository clean, portable, and secure.  
The inference code is fully functional and loads any compatible PyTorch `.pth` file.
