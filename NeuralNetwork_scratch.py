import pandas as pd
import numpy as np
import copy
np.random.seed(3)


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
            self.weights[str(1)] = np.random.randn(self.layers[-1], self.ip_layer_size)*0.01
        else:
            self.weights[str(self.num_of_layers)] = np.random.randn(self.layers[-1], self.layers[-2])*0.01

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
        self.cache["A0"] = data
        self.cache["Z0"] = data
        for i in range(0,len(self.layers)):
            A = A_next
            Z = np.dot(self.weights[str(i+1)], A) + self.biases[str(i+1)] #(neurons(i+1), m)
            A_next = self.activation(Z, self.layer_activations[i])  #(neurons(i+1), m)
            self.cache["A" + str(i+1)] = A_next
            self.cache["Z" + str(i+1)] = Z
        return A_next
    
    
    #Modify later 
    def calculate_cost(self, pred, true_label):
        #true and pred both are of the form (num of op layer vals x num of trainging examples)
        m = pred.shape[1]
        cost = pred*true_label
        cost = np.sum(cost, axis = 0)
        x = np.log(cost + 0.000000000001)
        cost = np.sum(x)
        return -cost/m
    
    # def calculate_cost(self, Y_hat, Y):
    #     m = Y.shape[1]
    #     L = -1./m * np.sum(Y * np.log(Y_hat))
    #     return L
    
    def gradient_checking(self, x, y, eps):
        grads_numerical = {}
        for i in range(self.num_of_layers):
            (a0, a1) = self.weights[str(i+1)].shape
            grads_numerical[str(i+1)] = np.zeros((a0, a1))
            for j in range(a0):
                for k in range(a1):
                    self.weights[str(i+1)][j][k] += eps;
                    O1 = self.forward_prop(x)
                    J1 = self.calculate_cost(O1, y)
                    #print("J1", J1)
                    self.weights[str(i+1)][j][k] -= 2*eps
                    O2 = self.forward_prop(x)
                    J2 = self.calculate_cost(O2, y)
                    self.weights[str(i+1)][j][k] +=eps
                    #print("J2", J2)
                    grad = (J1 - J2)/(2*eps)
                    grads_numerical[str(i+1)][j][k] = grad
        
        grad_biases_num = {}
        for i in range(0,self.num_of_layers):
            
            (b0, b1) = self.biases[str(i+1)].shape
            print("bo", b0, "b1", b1)
            grad_biases_num[str(i+1)] = np.zeros((b0, b1))
            for j in range(b0):
                self.biases[str(i+1)][j] += eps;
                O1 = self.forward_prop(x)
                J1 = self.calculate_cost(O1, y)
                #print("J1", J1)
                self.biases[str(i+1)][j]-= 2*eps
                O2 = self.forward_prop(x)
                J2 = self.calculate_cost(O2, y)
                self.biases[str(i+1)][j] +=eps
                #print("J2", J2)
                grad = (J1 - J2)/(2*eps)
                grad_biases_num[str(i+1)][j] = grad
            
        
        return grads_numerical, grad_biases_num

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
            
    
    def train(self, X, y, iterations, learning_rate):

        for i in range(iterations):
            for k in range(0,X.shape[1], 100):
                self.cache = {}
                self.gradients = {}
                x_ex = X[:, k:k+100]
                x_ex = np.reshape(x_ex, (X.shape[0], 100))
                output = self.forward_prop(x_ex)
                
                y_ex = y[: , k:k+100]
                y_ex = np.reshape(y_ex, (y.shape[0], 100))
                #print(output.shape, y[: , k].shape)
                
                # c = self.calculate_cost(output, y[:, k])
                # print("cost = ", c, "Iteration", i)
                self.back_prop(output, y_ex)
                for j in range(self.num_of_layers):
                    self.weights[str(j+1)] = self.weights[str(j+1)] - learning_rate*self.gradients["dW" + str(j+1)]
                    self.biases[str(j+1)] = self.biases[str(j+1)] - learning_rate*self.gradients["db" + str(j+1)]
                print("completed sgd", k)
            print(i, "completed")
            


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
x_test_flatten = x_test_flatten/255

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
model.add(512, "relu")
model.add(256, "relu")
#model.add(256, "relu")
model.add(10, "softmax")
base = model.forward_prop(x_train_flatten)

# cost_train ,x = model.calculate_cost(op, y_train_onehot)
# print (cost_train)
# model.back_prop(op, y_train_onehot)

model.train(x_train_flatten, y_train_onehot, 50, learning_rate=0.001)

op = model.forward_prop(x_train_flatten)
predicted_value = np.argmax(op, axis = 0)
print(accuracy(predicted_value, y_train))


# model = NeuralNetwork(5)
# model.add(4, "relu")
# model.add(3, "relu")
# model.add(4, "softmax")

# x_data = np.array([[1,6],[2,7],[3,9],[4,0],[5,1]])
# y_data = np.array([[0,0], [0,0], [1,0], [0,1]])

# op_og = model.forward_prop(x_data)

#model.train(x_data, y_data, 10)
# op = model.forward_prop(x_data)
# print(model.calculate_cost(op, y_data))
# calc_grad, calc_biases = model.gradient_checking(x_data, y_data, 0.001)
# model.back_prop(op, y_data)

# import copy

# weight_records = []
# dw_records_bp = []
# dw_records_num = []
# for i in range(1000):
#     model.cache = {}
#     model.gradients = {}
#     op = model.forward_prop(x_data)
#     print(model.calculate_cost(op, y_data))
#     model.back_prop(op, y_data)
#     dw_records_bp.append(copy.deepcopy(model.gradients))
#     calc_grad, calc_biases = model.gradient_checking(x_data, y_data, 0.001)
#     dw_records_num.append(copy.deepcopy(calc_grad))
#     dw_records_num.append(copy.deepcopy(calc_biases))
#     record_w = []
#     for j in range(model.num_of_layers):
#         model.weights[str(j+1)] = model.weights[str(j+1)] - 0.01*model.gradients["dW" + str(j+1)]
#         model.biases[str(j+1)] = model.biases[str(j+1)] - 0.01*model.gradients["db" + str(j+1)]
#         record_w.append(copy.deepcopy(model.weights[str(j+1)]))
#         record_w.append(copy.deepcopy(model.biases[str(j+1)]))
#     weight_records.append(record_w)
