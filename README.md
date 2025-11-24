# Image Recognition with Transfer Learning (CIFAR-10 Dataset)

This project implements transfer learning for image classification on the CIFAR-10 dataset using pre-trained models (VGG16, ResNet50, MobileNetV2) and compares performance with a custom CNN built from scratch.

## Overview

**Image Recognition with Transfer Learning (CIFAR-10 Dataset)**

This focuses on Neural Networks & Deep Learning, specifically exploring how transfer learning can reduce training time and improve performance compared to training models from scratch.

## Authors

- **Students**: Ainedembe Denis, Musinguzi Benson
- **Lecturer**: Harriet Sibitenda (PhD)

## Dataset

### CIFAR-10 Dataset
- **Source**: https://www.cs.toronto.edu/~kriz/cifar.html
- **Loading Method**: The dataset is automatically downloaded using TensorFlow's built-in loader (`tf.keras.datasets.cifar10.load_data()`)
- **Description**: 
  - 60,000 32x32 color images in 10 classes
  - 50,000 training images (5 batches of 10,000 images each)
  - 10,000 test images
  - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Image Format**: 32x32 RGB images (3 channels)
- **Automatic Download**: On first run, TensorFlow downloads the dataset (~170MB) from the official source and caches it in `~/.keras/datasets/` (or `C:\Users\<username>\.keras\datasets\` on Windows). Subsequent runs use the cached version.
- **Dataset Brief**: See `dataset/CIFAR-10 and CIFAR-100 datasets.pdf` or `dataset/readme.html` for detailed dataset information

### Dataset Structure
```
dataset/
├── readme.html
└── CIFAR-10 and CIFAR-100 datasets.pdf
```

## Dependencies

The project requires the following Python packages:

- **TensorFlow** (>=2.20.0) - Deep learning framework
- **NumPy** - Numerical computing
- **Matplotlib** - Plotting and visualization
- **Seaborn** - Statistical data visualization
- **scikit-learn** - Machine learning utilities (for evaluation metrics and data splitting)
- **Pandas** - Data manipulation and analysis
- **Jupyter Notebook** - Interactive development environment

### Installation

1. Install required packages:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pandas
```

Or install individually:
```bash
pip install notebook ipykernel
pip install tensorflow>=2.20.0
pip install numpy matplotlib seaborn scikit-learn pandas
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Navigate to `CIFAR10-TransferLearning.ipynb` in the browser

4. Clearing outputs via commandline:
```bash
jupyter nbconvert --clear-output --inplace CIFAR10-TransferLearning.ipynb
```

## Project Tasks

The project implements the following 9 tasks:

### Task 1: Load CIFAR-10 Dataset and Normalize Image Data
- Load CIFAR-10 dataset using TensorFlow's built-in loader (automatically downloads from official source)
- Visualize sample images from each class
- Normalize pixel values from [0, 255] to [0, 1]
- Convert labels to categorical (one-hot encoding)
- Split training data into train and validation sets (80-20 split)

### Task 2: Use Pre-trained Models for Transfer Learning
- Implement transfer learning using pre-trained models:
  - **VGG16**: Deep convolutional network with 16 layers
  - **ResNet50**: Residual network with 50 layers
  - **MobileNetV2**: Lightweight mobile-optimized network
- Upsample CIFAR-10 images from 32x32 to 224x224 (required input size for pre-trained models)
- Use ImageNet pre-trained weights

### Task 3: Freeze Lower Layers, Retrain Top Layers for 10 Epochs
- Freeze all base model layers (pre-trained weights remain unchanged)
- Add custom classification head:
  - GlobalAveragePooling2D
  - Dense layers (512, 256 units) with ReLU activation
  - Dropout layers (0.5, 0.3) for regularization
  - Final Dense layer (10 units) with softmax activation
- Train for 10 epochs with Adam optimizer (learning rate: 0.001)

### Task 4: Evaluate Accuracy and Confusion Matrix; Visualize Class Predictions
- Calculate test accuracy for all models
- Generate confusion matrices (normalized and raw)
- Visualize confusion matrices using heatmaps
- Display sample predictions with true labels, predicted labels, and confidence scores
- Generate classification reports with precision, recall, and F1-scores

### Task 5: Compare Performance with Custom CNN Built from Scratch
- Build custom CNN architecture:
  - 3 convolutional blocks (32, 64, 128 filters)
  - Batch normalization and dropout layers
  - Fully connected layers (512, 256, 10 units)
- Train on original 32x32 images (no upsampling required)
- Compare accuracy and training time with transfer learning models

### Task 6: Apply Data Augmentation and Measure Improvement
- Implement data augmentation techniques:
  - Random rotation (up to 15 degrees)
  - Random horizontal/vertical shifts (10% range)
  - Horizontal flipping
- Train custom CNN with augmented data
- Measure improvement in test accuracy
- Visualize augmented image examples

### Task 7: Plot Training History (Accuracy/Loss)
- Plot training and validation accuracy over epochs
- Plot training and validation loss over epochs
- Compare performance across all models
- Generate side-by-side comparison charts

### Task 8: Experiment with Fine-tuning Different Layers and Learning Rates
- Fine-tune models by unfreezing different numbers of layers (5, 10, 20 layers)
- Experiment with different learning rates (0.00001, 0.0001, 0.001)
- Compare validation accuracy across configurations
- Identify optimal fine-tuning strategy

### Task 9: Report and Analysis
- **Discussion**: Analyze how transfer learning reduces training time and improves performance
- Compare all model performances
- Summarize key findings:
  - Transfer learning advantages
  - Data augmentation impact
  - Fine-tuning considerations
  - Model comparison insights

## Project Structure

```
Image-Recognition-with-Transfer-Learning/
├── dataset/
│   ├── readme.html
│   └── Info-about-dataset.pdf
├── screenshots/
├── CIFAR10-TransferLearning.ipynb
├── README.md
└── .gitignore
```

## Key Features

- **Transfer Learning**: Leverage pre-trained ImageNet models (VGG16, ResNet50, MobileNetV2)
- **Custom CNN**: Build and train CNN from scratch for comparison
- **Data Augmentation**: Apply rotation, flipping, and shifting to improve generalization
- **Comprehensive Evaluation**: Accuracy, confusion matrices, classification reports
- **Visualization**: Training history plots, confusion matrices, prediction samples
- **Fine-tuning Experiments**: Systematic exploration of layer unfreezing and learning rates
- **Memory Optimization**: Efficient data generators for large-scale image processing

## Context

This project is part of the Intelligent Systems course assignment, focusing on:
- **Neural Networks & Deep Learning**: Practical application of transfer learning
- **Model Comparison**: Transfer learning vs. training from scratch
- **Performance Optimization**: Data augmentation and fine-tuning strategies
- **Comprehensive Analysis**: Evaluation metrics and visualization techniques