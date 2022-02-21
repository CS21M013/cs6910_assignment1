! pip install wandb
! wandb login
from Layer import *
from NeuralNetwork import *
from Helper import *
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import wandb
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

wandb.init(project="Assignment_1_random_randomSeed_SE", entity="cs21m007_cs21m013")
'''
Best Parameter obtained by Checking all Sweeps
'''
epochs = 10
acti='tanh'
lr = 1e-4
batch_size = 64
optimizer="Nadam"
init_type="Xavier"
loss_type="SquaredError"
reg=0
hidden_size=128
no_hidden_layer=4


'''
Testing
'''
model = Neural_Network()

for i in range(no_hidden_layer):
        model.add(Layer(hidden_size, activation=acti))

model.add(Layer(10, activation='softmax'))
print(model.layers)
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs, learning_rate=lr, optimizer=optimizer,val_split=0.1,init_type=init_type,loss_type=loss_type,reg=reg)
y_prob=model.predict(x_test)

help=Helper()
accuracy=help.accuracy(y_test,y_prob)

'''
Confusion Matrix logging
'''

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_test = x_test.reshape(x_test.shape[0], -1)
x_test = np.array(x_test/255., dtype=np.float32)
y_prob=np.empty(np.shape(y_test))
#finding y predicted
for i,x in enumerate(x_test):
    y_prob[i]= (model.predict(x)[0]).argmax()
    
print(y_test,y_prob.shape)

wandb.log({"conf_mat" : wandb.plot.confusion_matrix(preds=y_prob, y_true=y_test, class_names=class_list),"Test Accuracy": accuracy })