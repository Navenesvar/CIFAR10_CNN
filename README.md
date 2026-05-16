# CIFAR-10 Image Classification using Custom CNN

A deep learning image classification web application built using PyTorch and Streamlit. This project uses a custom Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset to classify images into 10 categories with over 90% training accuracy.

---

## Live Demo

🔗 Deployed App:

https://cifar10cnn-laccbhsvifex5wpcu9pzci.streamlit.app/

---

## Features

- Custom CNN architecture built from scratch
- CIFAR-10 dataset classification
- Data augmentation for better generalization
- Accuracy and loss tracking
- Streamlit web interface
- Image upload and prediction
- Confidence score display
- PyTorch implementation

---

## CIFAR-10 Classes

The model can classify images into the following categories:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

## Tech Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Matplotlib
- PIL

---

## Project Structure

```bash
cifar10_cnn/
│
├── app.py
├── model.py
├── advanced_cifar10_cnn.pth
├── requirements.txt
└── README.md
```

---

## Model Architecture

The custom CNN contains:

- Convolutional Layers
- Batch Normalization
- ReLU Activation
- Max Pooling
- Dropout Regularization
- Fully Connected Layers

### Architecture Flow

```text
Input Image (3x32x32)
        ↓
Conv2D + BatchNorm + ReLU
        ↓
Conv2D + BatchNorm + ReLU
        ↓
MaxPooling + Dropout
        ↓
Conv2D + BatchNorm + ReLU
        ↓
Conv2D + BatchNorm + ReLU
        ↓
MaxPooling + Dropout
        ↓
Conv2D + BatchNorm + ReLU
        ↓
Conv2D + BatchNorm + ReLU
        ↓
MaxPooling + Dropout
        ↓
Flatten
        ↓
Dense Layers
        ↓
Output (10 Classes)
```

---

## Training Details

### Optimizer

- Adam Optimizer

### Loss Function

- CrossEntropyLoss

### Techniques Used

- Batch Normalization
- Dropout
- Data Augmentation
- Learning Rate Scheduler
- Weight Decay

---

## Accuracy

| Metric | Result |
|---|---|
| Training Accuracy | 90% |
| Test Accuracy | 90% |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/cifar10-cnn.git
```

Move into project folder:

```bash
cd cifar10-cnn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
python -m streamlit run app.py
```

Open in browser:

```text
http://localhost:8501
```

---

## Streamlit Deployment

This application is deployed using Streamlit Community Cloud.

Live App:

https://cifar10cnn-laccbhsvifex5wpcu9pzci.streamlit.app/

---

## Sample Workflow

1. Upload an image
2. Model preprocesses image
3. CNN predicts class
4. Prediction and confidence score displayed

---

## Resume Description

Developed and deployed a deep learning-based image classification web application using PyTorch and Streamlit. Built an optimized custom CNN architecture with data augmentation, dropout, batch normalization, and learning rate scheduling to achieve over 90% accuracy on the CIFAR-10 dataset.

---

## Author

Navenesvar
