# Waste Classification Model with ResNet

This repository contains a deep learning-based waste classification model designed to automate the sorting of waste into three primary categories: **Recyclable**, **Compostable**, and **Trash**. The model leverages a fine-tuned **ResNet** architecture to achieve high accuracy and generalization.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Key Features](#key-features)
5. [Results](#results)
6. [Installation and Usage](#installation-and-usage)
7. [Future Improvements](#future-improvements)
8. [Acknowledgments](#acknowledgments)

---

## Introduction

Effective waste management is critical for sustainable environmental practices. This project focuses on automating the classification of waste into three categories:
- **Recyclable**: Paper, Plastic, Metal, Glass
- **Compostable**: Food Waste, Yard Waste
- **Trash**: Miscellaneous non-recyclable items

The model is designed to aid in reducing manual sorting efforts and increasing the accuracy of waste separation systems.

---

## Dataset

- **Name**: RealWaste Dataset
- **Structure**: Images are divided into 9 categories and mapped into 3 bins.
- **Preprocessing**:
  - Images were resized to **128x128 pixels**.
  - Normalized using `preprocess_input` to align with ImageNet mean and standard deviation.
  - Augmentation was applied for generalization.
- **Split**:
  - Training: 80%
  - Validation: 20%
  - Separate Test Dataset

---

## Model Architecture

### High-Level Overview
The model employs a **ResNet** (Residual Network) architecture pre-trained on ImageNet. The key components include:

1. **Feature Extraction**:
   - Leverages ResNet's convolutional layers to extract high-level features from the input images.
2. **Global Average Pooling (GAP)**:
   - Reduces feature dimensionality while preserving spatial information.
3. **Fully Connected Layers**:
   - Custom dense layers tailored for the classification task.
4. **Softmax Output**:
   - Provides probabilities for the three waste categories.

### Customizations
- **SGD Optimizer**: Chosen after extensive testing for optimal learning rate decay.
- **Class Weights**: Adjusted to address imbalances in the dataset.

---

## Key Features

- **Preprocessing**:
  - Augmentation techniques (rotation, flipping, zooming, etc.) for better generalization.
  - Image normalization using `preprocess_input`.
- **Class Weights**:
  - Weighted loss function to handle dataset imbalance effectively.
- **Visualization**:
  - Class Activation Maps (CAMs) to visualize the model's focus during predictions.

---

## Results

- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~96%
- **Test Accuracy**: **95.24%**
- **Test Loss**: **0.1350**

The model demonstrates excellent performance on unseen test data, with strong generalization capabilities.

---

## Installation and Usage

### Prerequisites
1. Python 3.x
2. TensorFlow >= 2.x
3. OpenCV for image processing

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/waste-classification-model.git
   cd waste-classification-model
   ```

2. **Prepare the Dataset**:
   Ensure the dataset is organized into training, validation, and test directories.

3. **Install Dependencies**:
   ```bash
   pip install tensorflow opencv-python
   ```

4. **Train the Model**:
   Use the provided training script:
   ```python
   from tensorflow.keras.models import load_model
   # Add training script here
   ```

5. **Test the Model**:
   ```python
   from tensorflow.keras.models import load_model
   import cv2
   import numpy as np
   from tensorflow.keras.applications.imagenet_utils import preprocess_input

   # Load model
   model = load_model('final_trashnet_transfer_learning_model.keras')

   # Load and preprocess image
   img_path = 'path/to/image.jpg'
   img = cv2.imread(img_path)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   img = cv2.resize(img, (128, 128))
   img = np.expand_dims(img, axis=0)
   img = preprocess_input(img)

   # Predict
   predictions = model.predict(img)
   predicted_class = np.argmax(predictions)
   print("Predicted Class Index:", predicted_class)
   ```

---

## Future Improvements

1. **Dataset Expansion**:
   - Incorporate additional waste categories to increase versatility.
2. **Higher Image Resolution**:
   - Train on larger image dimensions for more detailed feature extraction.
3. **Real-Time Deployment**:
   - Optimize the model for edge devices or IoT systems.

---

## Acknowledgments

- Dataset: [RealWaste Dataset](https://archive.ics.uci.edu/dataset/908/realwaste)
