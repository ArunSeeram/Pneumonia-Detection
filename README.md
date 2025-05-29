# 🩺 Pneumonia Detection Using Deep Learning 🧠

A machine learning project that detects **Pneumonia** from chest X-ray images using Convolutional Neural Networks (CNNs). This system aims to assist medical professionals in identifying pneumonia cases faster and more accurately.

---

## 📌 Project Overview

- **Goal**: Detect whether a patient has pneumonia from chest X-ray scans.
- **Dataset**: Chest X-Ray Images (Pneumonia) dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
- **Tech Stack**: Python, TensorFlow/Keras, CNN, OpenCV, Matplotlib.
- **Output**: Trained model with prediction script and performance metrics.

---

## 🧪 Features

- 🖼️ Preprocess chest X-ray images for input into CNN
- 🧠 Build and train a deep CNN model
- 📉 Plot training/validation loss and accuracy
- 📈 Evaluate on test data with accuracy, precision, recall, F1-score
- 📦 Save/load model for future use
- 🔍 Predict new X-ray image as "Normal" or "Pneumonia"

---

## 🧰 Tech Stack

| Tool           | Purpose                          |
|----------------|----------------------------------|
| Python         | Core programming language        |
| TensorFlow/Keras | Deep learning framework         |
| OpenCV         | Image processing                 |
| Matplotlib     | Plotting training graphs         |
| Scikit-learn   | Model evaluation metrics         |
| NumPy          | Array operations                 |

---

## 🗂️ Folder Structure
pneumonia-detection/
├── chest_xray/ # Dataset (train/test/val folders)
├── model/
│ └── pneumonia_model.h5 # Trained model (saved)
├── notebooks/
│ └── pneumonia_cnn.ipynb # Jupyter Notebook
├── predict.py # Script for single image prediction
├── train.py # Script to train the model
├── utils.py # Helper functions
├── README.md
└── requirements.txt
