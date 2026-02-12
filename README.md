# Neural Network from Scratch (Python)
## Overview

This project implements a fully connected neural network built entirely from scratch in Python using only NumPy. No deep learning frameworks (TensorFlow, PyTorch, or Keras) were used.

The objective of this project was to develop a deep understanding of neural network fundamentals by manually implementing forward propagation, backpropagation, and gradient-based optimization.

The model has been trained and evaluated on:

- **MNIST** (handwritten digit classification)
- **Fashion-MNIST** (clothing item classification)

---

## Architecture

The network is a standard feedforward neural network consisting of:

- Fully connected (dense) layers
- ReLU activation function for hidden layers
- Softmax activation function for the output layer
- Categorical Cross-Entropy loss function

The final layer outputs probability distributions across 10 classes.

---

## Implemented From Scratch

The following components were implemented manually:

- Weight initialization
- Forward propagation
- Backpropagation (gradient computation)
- Categorical Cross-Entropy loss
- Softmax activation
- Gradient descent parameter updates

All computations are fully vectorized using NumPy.


---

## Datasets

### MNIST  
28x28 grayscale images of handwritten digits (0–9).

### Fashion-MNIST  
28x28 grayscale images of clothing items across 10 categories.

All images are normalized and flattened into 784-dimensional input vectors before training.

---

## Tech Stack

- Python 3
- NumPy
- Matplotlib
- Pillow

---

## Purpose

This project was built to strengthen understanding of:

- Neural network training dynamics
- Optimization stability
- Activation behavior
- Loss surface behavior
- Model generalization

Building the network without high-level frameworks provided direct insight into how gradient-based learning works at a mathematical level.

---

## Future Improvements

- Implement advanced optimizers (Momentum / Adam)
- Add regularization (L2 / Dropout)
- Expand to convolutional neural networks (CNNs)
- Improve generalization performance on Fashion-MNIST
