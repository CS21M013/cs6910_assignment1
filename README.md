# Assignment_1_Feedforward_Neural_Networks
Authors: Anurag Mahendra Shukla (CS21M007) and Chandra Churh Chatterjee (CS21M013).

This repository contains Assignment 1 of CS6910

We have implemented Feedforward Neural Network from scratch and trained it for Fashion - Mnist dataset.

## File Descriptions:

### Assignment_1_FFN_fromScratch.ipynb
Jupyter Notebook containing feed-forward neural network implemntation and hyperparameter tuning using Wandb sweeps.

### Assignment_1_FFN_MNIST.ipynb
Jupyter Notebook containing model trained for 3 recommended parameters for MNIST dataset. (Answer 10)

### Layer.py
Contains Layer class which is the building block for a single layer

### NeuralNetwork.py
Contains NeuralNetwork Class which uses the Layer class to build the NeuralNetwork architecture. Also contains functions for forward propagation, backward propagation, fitting/training, predictions and various optimizations.

### Helper.py
Contains Helper functions for creating batches, computing loss and accuracy.

### Individual_model_testing.py
Contains example of a simple model training.

### TestingandConfusionMatrix.py
Code for Testing and generating confustion matrix IN WANDB.

### ClassesPlot.py
Code for Ploting one sample image from each class of Fashion Mnist data (Answer 1)

### WandbSweep.py
Contains code for tuning hyperparameters using wandb sweep.


