# Feedforward Neural Networks from scratch
Authors: Anurag Mahendra Shukla (CS21M007) and Chandra Churh Chatterjee (CS21M013).

Report Link: https://wandb.ai/cs21m007_cs21m013/Assignment_1_random_randomSeed/reports/CS6910-Assignment-1-Feed-Forward-Neural-Network-from-scratch-for-Fashion-MNIST-classification---VmlldzoxNTk0NzEz?accessToken=yq11khq5xset2zmriit9v16x0ijt80ycyaw33ewod2qb7z6yfhq2pm42tmnminvg

This repository contains Assignment 1 of CS6910 (Deep Learning)

We have implemented Feedforward Neural Network from scratch and trained it for Fashion - MNIST dataset. The report link mentioned above represents
the entire experimentation including hyper parameter tuning for Fashion - MNIST dataset and using ideas from the experiment to test the performance of 
different combinations of hyper-parameters on the MNIST data. The repository also allows one to use wandb for tuning the model and better visualization

## File Descriptions:

### Assignment_1_FFN_fromScratch.ipynb
Jupyter Notebook containing feed-forward neural network implemntation and hyperparameter tuning using Wandb sweeps.

### Assignment_1_FFN_MNIST.ipynb
Jupyter Notebook containing model trained and tested for 3 recommended hyper-parameters for MNIST dataset. (Answer 10)

### Assignment_1_FFN_Fashion_MNIST.ipy
Jupyter Notebook containing training and testing of model with best hyperparameters for Fashion-Mnist dataset.

### Assignment_1_FFN_fromScratch_without_outputs.ipynb
Jupyter Notebook containing only code for feedforward neural networks and hyper-parameter tuning without outputs.

(Because the sweep outputs were very long, we have made a separate notebook just for the code. The sweep outputs can be still found in Assignment_1_FFN_fromScratch.ipynb)

## Installing and logging into wandb :

If you haven't installed wandb, you can do it using

```
! pip install wandb
! wandb login <your wandb authentication key here>
```

in your jupyter notebook

or 

```
pip install wandb
```

```
wandb login
```

in command line

### Initializing wandb project

```
entity="cs21m007_cs21m013"
project="checking"
wandb.init(project=project, entity=entity)
```

## Initializing the structure of your model

### Initializing your model
```
model = Neural_Network()
```

### Adding layers to your model
```
model.add(Layer(Num_of_hidden_units, activation=activation_function))
```

Where activation_function = 'tanh', 'relu', 'sigmoid' or 'softmax'

## Fitting the model
### Set the hyper-parameter configuration

```
epochs = 10
lr = 1e-4
batch_size = 64
optimizer="RMSprop"
init_type="Xavier"
loss_type="CrossEntropy"
reg=0.0005
```


Note: You can set your own hyper-parameters for training

### Call the fit function


```
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs, learning_rate=lr, optimizer=optimizer,val_split=0.1,init_type=init_type,loss_type=loss_type,reg=reg)
```

x_train: Your training set input
y_train: Your training set output

## Prediction using the model

```
y_pred=model.predict(x_test)
```

### Calculating Test accuracy

````
help=Helper()
accuracy=help.accuracy(y_test,y_prob)
````

## Hyper-parameter tuning using wandb

### Creating a sweep Configuration

```
sweep_config = {
    'method': 'random', #bayes, random, grid methods can be used for tuning.
    'metric': {
      'name': 'Val/Accuracy', # goal is to maximize the validation accuracy.
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
```

### Initializing project in wandb

``` 
sweep_id = wandb.sweep(sweep_config, entity=entity, project=project) 
```

You will get a sweep URL which you can visit to see the results of sweep.

### Sweep for hyperparameter search

```
sweep_count = 100   # You can change this
wandb.agent(sweep_id, train, count= sweep_count)
```

`Note: This will take about 6-7 hours for default(100) sweep counts`


