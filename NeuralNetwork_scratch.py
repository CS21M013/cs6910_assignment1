import pandas as pd
import numpy as np
#np.random.seed(1)


class NeuralNetwork:
    def __init__(self, input_layer):
        self.ip_layer_size = input_layer
        self.layers = []
        self.layer_activations = []
        self.parameters = {}
        self.biases = {}
        self.weights = {}
        self.num_of_layers = 0
        self.cache = {}
        self.gradients = {}

    def add(self, num_of_units, activation):
        self.layers.append(num_of_units)
        self.layer_activations.append(activation)
        self.num_of_layers = self.num_of_layers + 1
        if(self.num_of_layers == 1):
            self.weights[str(1)] = np.random.randn(self.layers[-1], self.ip_layer_size)
        else:
            self.weights[str(self.num_of_layers)] = np.random.randn(self.layers[-1], self.layers[-2])

        self.biases[str(self.num_of_layers)] = np.zeros((num_of_units, 1))

    def activation(self, Z , function):
        ##Do something
        if(function == "relu"):
            return Z*(Z>0)
        elif(function == "sigmoid"):
            return (1/(1 + np.exp(-Z)))
        elif(function == "tanh"):
            pass
        elif(function == "softmax"):
            maxx = np.max(Z, axis = 0, keepdims=True)
            #print(maxx.shape)
            Z = Z - maxx;
            Z = np.exp(Z)
            sum_Z = np.sum(Z, axis = 0, keepdims=True)
            Z = Z/sum_Z
            return Z
        else:
            raise ValueError("Please check function argument. Supported activations: 'relu', 'sigmoid', 'tanh', 'softmax' ")
            
    def activation_grad(self, Z, function):
        if(function == "sigmoid"):
            A = self.activation(Z, "sigmoid")
            return A*(1-A)
        elif(function == "relu"):
            return (Z>0)*1
        elif(function == "tanh"):
            pass
        else:
            raise ValueError("Please check function argument for derivative function. Supported activations: 'relu', 'sigmoid', 'tanh'")
    
    def forward_prop(self, data):
        #data is of dimension (num_of_parameters x num_of_training_examples(m))
        A_next = data
        self.cache["A0"] =data
        self.cache["Z0"] = data
        for i in range(0,len(self.layers)):
            A = A_next
            Z = np.dot(self.weights[str(i+1)], A) + self.biases[str(i+1)] #(neurons(i+1), m)
            A_next = self.activation(Z, self.layer_activations[i])  #(neurons(i+1), m)
            self.cache["A" + str(i+1)] = A_next
            self.cache["Z" + str(i+1)] = Z
        return A_next
    
    
    Modify later 
    def calculate_cost(self, pred, true_label):
        #true and pred both are of the form (num of op layer vals x num of trainging examples)
        cost = pred*true_label
        cost = np.sum(cost, axis = 0)
        x = np.log(cost + 0.000000000001)
        cost = np.sum(x)
        return -cost
    
        


    def back_prop(self, pred, true_label):
        #true and pred both are of the form (num of op layer vals x num of trainging examples)
        m = pred.shape[1]
        dZ = pred - true_label # (10, m)
        for k in range(self.num_of_layers, 0, -1):
            dW = np.dot(dZ, self.cache["A" + str(k-1)].T)/m
            #(neurons(k), m) dot (m, neurons(layer k-1)) = (neurons(k), neurons(k-1))
            db = np.sum(dZ, axis = 1, keepdims = True)/m # (neurons(k))
            if(k>1):
                dA = np.dot(self.weights[str(k)].T, dZ)
                # (neurons(k-1), neurons(k)) dot (neurons(k), m)) = (neurons(k-1), m)
                dZ = dA*(self.activation_grad(self.cache["Z" + str(k-1)], self.layer_activations[k-2]))
            self.gradients["dW" + str(k)] = dW
            self.gradients["db" + str(k)] = db
    
    def train(self, X, y, iterations, learning_rate = 0.01):
        for i in range(iterations):
            self.cache = {}
            self.gradients = {}
            output = self.forward_prop(X)
            c = self.calculate_cost(output, y)
            print("cost = ", c)
            self.back_prop(output, y)
            for j in range(self.num_of_layers):
                self.weights[str(j+1)] = self.weights[str(j+1)] - learning_rate*self.gradients["dW" + str(j+1)]
                self.biases[str(j+1)] = self.biases[str(j+1)] - learning_rate*self.gradients["db" + str(j+1)]
            


def accuracy(pred, true_val):
    count = 0
    for i in range(pred.shape[0]):
        if(pred[i] == true_val[i]):
            count+=1
    return count/pred.shape[0]

    # def train(bactch, op):
    #     n = traingsize/bs
    #     for i in range(n):
    #         feed
    #         grads = bp
    #         if(op ==):
    #             call function for optimizer
    #30,70
        


            
        
        
    

        

dataset = pd.read_csv('archive/fashion-mnist_train.csv')
x_train = dataset.iloc[:, 1:].values
y_train = dataset.iloc[:, 0].values
x_train_flatten = np.reshape(x_train, (60000, 784)).T
x_train_flatten = x_train_flatten/255

dataset = pd.read_csv('archive/fashion-mnist_test.csv')
x_test = dataset.iloc[:, 1:].values
y_test = dataset.iloc[:, 0].values
x_test_flatten = np.reshape(x_test, (10000, 784)).T


# initializing variables

num_of_inputs = 784
num_of_labels = 10
num_training_examples = 60000
num_test_examples = 10000

#One hot encoding for output

y_train_onehot = np.zeros((num_of_labels, num_training_examples))
for i in range(num_training_examples):
    y_train_onehot[y_train[i], i] = 1

y_test_onehot = np.zeros((num_of_labels, num_test_examples))
for i in range(num_test_examples):
    y_test_onehot[y_test[i], i] = 1


#print(x_train_flatten.shape, x_test_flatten.shape)


model = NeuralNetwork(784)
model.add(20, "relu")


model.add(10, "softmax")
#op = model.forward_prop(x_train_flatten)

# cost_train ,x = model.calculate_cost(op, y_train_onehot)
# print (cost_train)
# model.back_prop(op, y_train_onehot)

model.train(x_train_flatten, y_train_onehot, 500)

op = model.forward_prop(x_train_flatten)
predicted_value = np.argmax(op, axis = 0)
print(accuracy(predicted_value, y_train))