# Pneumonia-Classification with XAI
This project applies deep learning and Explainable AI (XAI) techniques to classify chest X-ray images into pneumonia and normal categories. It uses both a custom Convolutional Neural Network (CNN) and transfer learning with the VGG16 model, integrating interpretability tools like Grad-CAM, Saliency Maps, and SmoothGrad to enhance transparency and trust in medical AI systems.

---
## 📌 Project Objectives
- To develop a pneumonia classification model to classify chest X-ray images into categories of pneumonia- present or normal.
- Preprocess X-ray images for optimized model performance by preprocessing steps include resizing, data augmentation, and normalization.
- Evaluate the performance of the machine learning model by evaluating the model's accuracy, precision, recall, f1-score to ensure high accuracy in classificiation.
- Incorporate explainable artificial intelligence (XAI) to highlight abnormal areas.

---
## Dataset
- Source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---
## Methodology 
Preprocessing: 
- Resize images to 150×150
- Augmentation (rotation, flipping, zoom, brightness)
- Z-score normalization

Models:
1. Custom CNN
   - 3 convolutional layers (ReLU + MaxPooling)
   - Fully connected layers + Dropout
   - Sigmoid output for binary classification
  
2. Transfer learning with VGG16
   - Pre-trained on ImageNet
   - Top layers removed (include_top=False)
   - Custom dense layers with dropout
   - Output layer with softmax for 2-class classification

 ---
 ## Results
 | Model | Accuracy | Precision | Recall    | F1-Score |
| ----- | -------- | --------- | --------- | -------- |
| CNN   | 94.97%   | 0.94/0.97 | 0.97/0.93 | 0.95     |
| VGG16 | 96.72%   | 0.97/0.97 | 0.97/0.97 | 0.97     |

---
## Explainable AI (XAI)
XAI methods applied:
- Grad-CAM: Visual heatmaps highlighting decision regions
- Saliency Map: Gradient-based visualization of important pixels

---
## User Interface
<img width="580" height="608" alt="image" src="https://github.com/user-attachments/assets/13eeae32-6743-49f4-92b0-509f2146ec8a" />



