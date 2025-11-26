# Image Recognition with Transfer Learning (CIFAR-10 Dataset)

This project implements transfer learning for image classification on the CIFAR-10 dataset using a pre-trained MobileNetV2 model and compares performance with a custom CNN built from scratch.

## Overview

**Image Recognition with Transfer Learning (CIFAR-10 Dataset)**

This focuses on Neural Networks & Deep Learning, specifically exploring transfer learning with MobileNetV2 and comparing performance with a custom CNN built from scratch. The project includes comprehensive analysis with dynamic evaluation metrics that adapt based on actual results.

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
- Visualize sample images from each class (2x5 grid)
- Normalize pixel values from [0, 255] to [0, 1]
- Split training data into train and validation sets (80-20 split with stratification)
- Dataset split: 40,000 training, 10,000 validation, 10,000 test images

### Task 2: Use Pre-trained Models for Transfer Learning
- Implement transfer learning using **MobileNetV2**: Lightweight mobile-optimized network
- Upsample CIFAR-10 images from 32x32 to 96x96 (optimized size for MobileNetV2, smaller than standard 224x224)
- Use ImageNet pre-trained weights
- Build efficient `tf.data` pipelines with data augmentation and prefetching

### Task 3: Freeze Lower Layers, Retrain Top Layers for 10 Epochs
- Freeze all base model layers (pre-trained weights remain unchanged)
- Add custom classification head:
  - GlobalAveragePooling2D
  - Dropout layer (0.3) for regularization
  - Final Dense layer (10 units) with softmax activation
- Train for 10 epochs with Adam optimizer (learning rate: 0.001)
- Track training progress with custom `DatasetCountCallback` that displays:
  - Dataset size and batch information at the start of each epoch
  - Completion message showing images processed per epoch
  - Comprehensive summary with performance metrics, overfitting analysis, and training efficiency

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
  - **Transfer Model**: Random rotation (up to 15 degrees) and horizontal flipping
  - **Custom CNN**: Horizontal flipping and random shifts (using resize_with_pad and random_crop)
- Build separate `tf.data` pipelines for transfer learning and custom CNN
- Train models with augmented data
- Measure improvement in test accuracy
- Visualize augmented image examples (original vs augmented comparisons)

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
- **Comprehensive Results Analysis**: Dynamic analysis with conditional logic based on actual performance metrics
  - Test Set Accuracy comparison
  - Validation Accuracy analysis
  - Overfitting Analysis (train vs validation gap)
  - Training Configuration summary
  - Model Complexity comparison
- **Key Insights & Conclusions**: Markdown notes section with findings and recommendations
- **Discussion**: Analyze how transfer learning performs compared to custom CNN
- Compare all model performances with adaptive narrative based on results

## Project Structure

```
Image-Recognition-with-Transfer-Learning/
├── dataset/
│   ├── readme.html
│   └── CIFAR-10 and CIFAR-100 datasets.pdf
├── Presentation Slides/
│   ├── Image Recognition with Transfer Learning (CIFAR-10 Dataset).pdf
│   └── Image Recognition with Transfer Learning (CIFAR-10 Dataset).pptx
├── screenshots/
│   ├── Custom CNN.png
│   ├── Transfer (Frozen Base).png
│   └── Transfer (Fine-tuned).png
├── .vscode/
│   └── settings.json
├── CIFAR10-TransferLearning.ipynb
├── README.md
└── .gitignore
```

## Key Features

- **Transfer Learning**: Leverage pre-trained MobileNetV2 model with ImageNet weights
- **Custom CNN**: Build and train CNN from scratch for comparison
- **Data Augmentation**: Apply rotation (15°), horizontal flipping, and image shifts to improve generalization
- **Efficient Data Pipelines**: `tf.data` pipelines with batching, prefetching, and parallel processing
- **Training Progress Tracking**: Custom callback that displays dataset counts, batches, and performance metrics per epoch
- **Comprehensive Evaluation**: Accuracy, confusion matrices, classification reports
- **Dynamic Results Analysis**: Conditional analysis that adapts narrative based on actual performance values
- **Visualization**: Training history plots, confusion matrices, prediction samples
- **Fine-tuning Experiments**: Unfreeze base model layers and fine-tune with lower learning rates
- **Memory Optimization**: Efficient data generators for large-scale image processing

## Results Analysis

The notebook includes a comprehensive results analysis section with dynamic conditional logic:

1. **Test Set Accuracy**: Compares generalization performance with adaptive categorization
2. **Validation Accuracy**: Analyzes model selection metrics with performance level categorization
3. **Overfitting Analysis**: Evaluates train-validation gaps with severity categorization
4. **Training Configuration**: Summarizes epochs and training efficiency
5. **Model Complexity**: Compares parameter counts and efficiency
6. **Key Insights & Conclusions**: Markdown notes section with findings and recommendations

All analysis sections use unified conditional functions that adapt the narrative based on actual performance values, providing context-aware insights rather than static descriptions.

## Context

This project is part of the Intelligent Systems course assignment, focusing on:
- **Neural Networks & Deep Learning**: Practical application of transfer learning
- **Model Comparison**: Transfer learning vs. training from scratch
- **Performance Optimization**: Data augmentation and fine-tuning strategies
- **Comprehensive Analysis**: Dynamic evaluation metrics and visualization techniques
- **Real-world Insights**: Understanding when transfer learning works and when custom architectures are better