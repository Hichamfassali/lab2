# MNIST Classification and Object Detection with Pytorch

## Objective:
The goal of this project is to familiarize with the Pytorch library by building and comparing various neural architectures for computer vision tasks. Specifically, we will implement and compare CNN, Faster R-CNN, and fine-tuned models (VGG16 and AlexNet) on the MNIST dataset.

---

## Tasks:

### Part 1: CNN Classifier

1. **Dataset: MNIST**
   - The MNIST dataset is a collection of 28x28 grayscale images of handwritten digits (0-9). You can download the dataset from the following sources:
     - [Kaggle - MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
     - [Official MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

2. **Establish a CNN Architecture for MNIST Classification**
   - Using Pytorch, create a Convolutional Neural Network (CNN) architecture to classify digits from the MNIST dataset. Your CNN should include:
     - **Convolutional Layers**: Define multiple convolution layers.
     - **Pooling Layers**: Use max pooling to reduce dimensionality.
     - **Fully Connected Layers**: After the convolutional layers, define a few fully connected layers to classify the digits.
     - **Hyperparameters**: Set the values for kernels, padding, stride, optimizers, and regularization (like dropout).
     - **Run Model on GPU**: Ensure the model is run on GPU (if available).

3. **Implement Faster R-CNN**
   - Use the Faster R-CNN architecture to perform object detection on the MNIST dataset. Faster R-CNN is a region-based CNN that incorporates an object detection network.
   - Use a pretrained Faster R-CNN model available in Pytorch and fine-tune it for the MNIST dataset.

4. **Comparison of the Two Models (CNN vs Faster R-CNN)**
   - Compare the following metrics for both models:
     - **Accuracy**
     - **F1 Score**
     - **Loss**
     - **Training Time**

### Part 2: Fine-Tuning with Pretrained Models

5. **Fine-Tune Pretrained Models (VGG16 & AlexNet)**
   - Use pretrained models like VGG16 and AlexNet for transfer learning and fine-tune them on the MNIST dataset.
   - Compare the performance of these models with the CNN and Faster R-CNN models based on the same metrics (Accuracy, F1 Score, Loss, Training Time).



