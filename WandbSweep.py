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

sweep_config = {
    'method': 'random', #grid, random
    'metric': {
      'name': 'Val/Accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'epochs': {
            'values': [5,10]
        },
        'no_hidden_layer':{
            'values': [3,4,5]  
        },
        'learning_rate': {
            'values': [1e-3,1e-4]
        },
        'opt': {
            'values': ['gd','sgdm','nesterov','rmsprop','adam','nadam']
        },
        'activation': {
            'values': ['relu', 'sigmoid','tanh']
        },
        'batch_size':{
            'values':[16,32,64]
        },
        'size_hidden':{
            'values':[32,64,128]
        },
        'reg':{
            'values': [0,0.0005,0.5]
        },
        'init_type':{
            'values': ['Xavier','Random']  
        }
    }
}

sweep_id = wandb.sweep(sweep_config, entity="cs21m007_cs21m013", project="Assignment_1_random_randomSeed_SE")

def train():
    steps = 0
    # Default values for hyper-parameters we're going to sweep over
    config_defaults = {
        'epochs': 5,
        'no_hidden_layer':3,
        'learning_rate': 1e-3,
        'opt':'adam',
        'activation':'sigmoid',
        'batch_size':16,
        'size_hidden':32,
        'reg':0,
        'init_type':'Xavier'
    }

    # Initialize a new wandb run
    wandb.init(project='Sweep_test', entity='cs21m007_cs21m013',config=config_defaults)
    
    
    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config
    lr = config.learning_rate
    epochs = config.epochs
    opt = config.opt
    acti=config.activation
    batch_size = config.batch_size
    hidden_size=config.size_hidden
    reg=config.reg
    init_type=config.init_type
    no_hidden_layer=config.no_hidden_layer
    if opt=="gd":
        opt="GD"
    elif opt=='adam':
      opt="Adam"
    elif opt=='rmsprop':
      opt="RMSprop"
    elif opt=='sgdm':
      opt='SGDM'
    elif opt=='nadam':
      opt="Nadam"
    elif opt=='nesterov':
      opt="Nesterov"
    # Model training here
    model = Neural_Network()
    for i in range(no_hidden_layer):
        model.add(Layer(hidden_size, activation=acti))

    model.add(Layer(10, activation='softmax'))
    print(model.layers)
    model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs, learning_rate=lr, optimizer=opt,val_split=0.1,init_type=init_type,loss_type="SquaredError",reg=reg)
    
wandb.agent(sweep_id, train,count=100)