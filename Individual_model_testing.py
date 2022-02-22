! pip install wandb
! wandb login
from Layer import *
from Helper import *
from NeuralNetwork import *
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import wandb
def one_hot(Y):
    num_labels = len(set(Y))
    new_Y = []
    for label in Y:
        encoding = np.zeros(num_labels)
        encoding[label] = 1.
        new_Y.append(encoding)
    return np.array(new_Y)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
print(x_train.shape, x_test.shape)
x_train = np.array(x_train/255., dtype=np.float32)
x_test = np.array(x_test/255., dtype=np.float32)
  
y_train = one_hot(y_train)
y_test = one_hot(y_test)
print(y_train.shape, y_test.shape)

'''
Paramter Setting
'''
epochs = 5
lr = 1e-3
batch_size = 16
optimizer="GD"
init_type="Xavier"
loss_type="CrossEntropy"
reg=0
no_hidden_layer=3
size_hidden=64

'''
Model Training
'''

model = Neural_Network()
for i in range(no_hidden_layer):
        model.add(Layer(hidden_size, activation=acti))
model.add(Layer(10, activation='softmax'))
print(model.layers)
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs, learning_rate=lr, optimizer=optimizer,val_split=0.1,init_type=init_type,loss_type=loss_type,reg=reg)