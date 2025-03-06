# MNIST Digit Classification with CNN

📌 **Project Overview**

This project is a deep learning-based handwritten digit recognition system using Convolutional Neural Networks (CNNs) on the MNIST dataset. The model is trained to classify digits (0-9) from grayscale images of handwritten numbers.

📂 Dataset: MNIST

60,000 training images & 10,000 test images

Each image is 28x28 pixels (grayscale)

Contains handwritten digits from 0 to 9

A widely used benchmark dataset for image classification

🛠 **Tech Stack**

Python

TensorFlow/Keras (for deep learning)

NumPy (for data manipulation)

Matplotlib (for visualization)

Scikit-learn (for dataset handling)

🚀 **Model Architecture**

The CNN model consists of:

Convolutional Layers: Extracts features from input images

MaxPooling Layers: Reduces dimensions while preserving important features

Dropout Layer: Prevents overfitting

Fully Connected Layer: Makes predictions based on extracted features

Softmax Output Layer: Classifies the images into 10 digit classes (0-9)

📜 **Code Explanation**

The project involves importing necessary libraries, loading and preprocessing the dataset, and building a CNN model. The model includes convolutional layers for feature extraction, max-pooling layers for dimensionality reduction, and dropout layers to prevent overfitting. After training the model on the MNIST dataset, it is evaluated for accuracy and used to make predictions on test images.

📊 **Visualizing Training Performance**

The training history, including accuracy and loss curves, is plotted to analyze the model's performance. This helps in understanding how well the model generalizes to unseen data.

📌 **Future Improvements**

✅ Improve accuracy by adding more layers and filters
✅ Train for more epochs for better generalization
✅ Use custom handwritten digits for testing
✅ Deploy the model using Flask for a web-based classification tool

📜 **License**

This project is open-source and available under the MIT License.


