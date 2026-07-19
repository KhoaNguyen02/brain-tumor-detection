# Brain Tumor Classification App

## Overview
Welcome to the **Brain Tumor Classification App**! This application implements state-of-the-art deep learning models to classify brain tumors from fMRI/MRI images into four categories:
- **No Tumor**
- **Glioma**
- **Meningioma**
- **Pituitary Tumor**

> [!WARNING]
> 
> This tool is intended for demonstration purposes only. It should not be used as a substitute for professional medical diagnosis.

## Features
- **Model Selection:** Choose from **CNN**, **ResNet**, **DenseNet**, or an **Auto** mode that combines predictions from all three models.
- **Interactive UI:** Upload an image and get instant predictions with a detailed confidence breakdown.
- **Visualization:** View the uploaded image alongside prediction statistics for better understanding.

## Installation

To get started with the Brain Tumor Classifier, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/KhoaNguyen02/brain-tumor-detection.git
    cd brain-tumor-classifier
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application:**
    ```bash
    streamlit run app.py
    ```

## Usage
Once the app is running, follow these steps to use it:

1. **Select a Model:** Choose between **CNN**, **ResNet**, **DenseNet**, or **Auto** mode for predictions.
2. **Upload an Image:** Upload an fMRI/MRI image in JPG, JPEG, or PNG format.
3. **View Results:** The app will display the prediction, confidence level, and a probability breakdown for each condition.

## Models
### CNN
- **Architecture:** Custom Convolutional Neural Network
- **Use Case:** Lightweight and fast model with decent accuracy.

### ResNet
- **Architecture:** Residual Network (ResNet-101)
- **Use Case:** Deeper network with better accuracy, suitable for more complex images.

### DenseNet
- **Architecture:** Densely Connected Convolutional Networks (DenseNet-121)
- **Use Case:** Highly accurate but computationally intensive.

### Auto Mode
- **Ensemble Approach:** Combines the strengths of CNN, ResNet, and DenseNet by weighting their predictions to give a more robust final output.