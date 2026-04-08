![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-purple)
![Architecture](https://img.shields.io/badge/Architecture-CNN%20%7C%20ResNet18-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

# Facial Emotion Recognition System

Facial Emotion Recognition System is a computer vision project focused on multi-class facial expression classification using the FER-2013 dataset and a modular PyTorch-based workflow.

This repository is designed to demonstrate production-style ML engineering practices for image classification, including reproducible training structure, evaluation outputs, inference utilities, and a clearer separation between experimentation and application logic.



This project focuses on classifying facial expressions such as:

- happy
- neutral
- surprise
- sad
- angry
- fear
- disgust

The repository is valuable as a portfolio piece because it does more than train a model. It also presents:

- structured training workflows
- multiple model backbones
- evaluation outputs
- inference tooling
- artifact management
- modular project organization
- explicit limitations and ethical considerations

That makes it a stronger engineering signal than a notebook-only vision project.



## What is implemented today

The repository currently includes:

- a PyTorch-based emotion classification workflow
- support for a lightweight CNN model
- support for a ResNet18-based model
- training and evaluation scripts
- saved artifact paths for model checkpoints
- classification report and confusion matrix outputs
- inference utilities for image-based prediction
- test files and package structure improvements
- fixes for imports, dependency consistency, and execution reliability

This creates a stronger reviewer experience because the project is organized as a working ML repository rather than only a demo.



## Architecture

```text
Input image
    ↓
Preprocessing
    ↓
CNN or ResNet18 model
    ↓
Logits
    ↓
Prediction / evaluation pipeline
    ↓
Metrics artifacts and inference output

