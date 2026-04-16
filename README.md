<h1 align="center">
✨🚀 Bidabi — Machine Learning Pipeline 🚀✨
</h1>

<p align="center">
  <img src="https://media.giphy.com/media/3o7TKtnuHOHHUjR38Y/giphy.gif" width="200">
</p>

<p align="center">
  <b>End-to-End Machine Learning Pipeline with PyTorch & DVC</b>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue">
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-red">
  <img src="https://img.shields.io/badge/DVC-DataVersioning-green">
</p>
![Typing SVG](https://readme-typing-svg.herokuapp.com?color=00F7FF&lines=End-to-End+ML+Pipeline;PyTorch+%7C+DVC+%7C+Git)

---

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![DVC](https://img.shields.io/badge/DVC-DataVersioning-green)

---

## 📖 Overview

This project implements a complete and reproducible machine learning pipeline for image classification.

The goal is to build a system capable of recognizing food products (such as bread, milk, and butter) from images.

The project follows an end-to-end workflow including data preparation, dataset versioning, model training, and reproducibility using MLOps practices.

---

## ⚙️ Project Structure
.
├── data/
│ └── raw/ # RAW dataset (versioned with DVC)
├── src/
│ ├── asyscrapper.py # Data collection script
│ └── train.py # Training pipeline
├── model.pth # Trained model (versioned with DVC)
├── requirements.txt
└── README.md


---

## 🔄 Machine Learning Pipeline

The pipeline includes the following steps:

- Dataset creation and structuring (image classification dataset)
- Data versioning using DVC
- Splitting dataset into train / validation / test sets (70/20/10)
- Data augmentation (resize, horizontal flip, tensor conversion)
- Training a deep learning model (ResNet18)
- Monitoring training and validation loss
- Saving the best trained model

---

## 📦 Dataset

- Dataset format:
- 
---

## 🔄 Machine Learning Pipeline

The pipeline includes the following steps:

- Dataset creation and structuring (image classification dataset)
- Data versioning using DVC
- Splitting dataset into train / validation / test sets (70/20/10)
- Data augmentation (resize, horizontal flip, tensor conversion)
- Training a deep learning model (ResNet18)
- Monitoring training and validation loss
- Saving the best trained model

---

## 📦 Dataset

- Dataset format: 

- Categories:
- bread
- milk
- butter

- The dataset is:
- structured in RAW format
- versioned using DVC
- reproducible across environments

---

## 🤖 Model

- Model: ResNet18 (pretrained on ImageNet)
- Task: Image classification
- Output file: `model.pth`
- Versioning: handled using DVC (`model.pth.dvc`)

The model is trained on the dataset and saved automatically during training.

---

## 🔁 Reproducibility

To reproduce the project:

```bash
git clone <your-repo-url>
cd bidabi-clone-alone

pip install -r requirements.txt
dvc pull

python src/train.py



