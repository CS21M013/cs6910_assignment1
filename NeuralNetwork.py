import numpy as np
from Helper import *
from Layer import *
from sklearn.model_selection import train_test_split
class Neural_Network:
    def __init__(self):
        self.layers = dict()
        self.cache = dict()
        self.grads = dict()
        
    def add(self, layer):
        self.layers[len(self.layers)+1] = layer

    def forward(self, x, init_type="Xavier"):
        for idx, layer in self.layers.items():

            layer.input = np.array(x, copy=True)
            if layer.W is None:
                layer.initialize_params(layer.input.shape[-1], layer.hidden_units,init_type)

            layer.Z = x @ layer.W + layer.b
        
            if layer.activation is not None:
                layer.A = layer.activation_fn(layer.Z)
                x = layer.A
            else:
                x = layer.Z
            #x = layer.forward(x)
            self.cache[f'W{idx}'] = layer.W
            self.cache[f'Z{idx}'] = layer.Z
            self.cache[f'A{idx}'] = layer.A
        return x

    def backward(self, y, loss_type,reg=0):
        last_layer_idx = max(self.layers.keys())
        m = y.shape[0]
        for idx in reversed(range(1, last_layer_idx+1)):
            if idx == last_layer_idx:
                if loss_type=="CrossEntropy":
                    self.grads[f'dZ{idx}'] = self.cache[f'A{idx}'] - y
                elif loss_type=="SquaredError":
                    self.grads[f'dZ{idx}'] = (self.cache[f'A{idx}'] - y) * self.layers[idx].activation_fn(self.cache[f'Z{idx}'], derivative=True)
            else:
                self.grads[f'dZ{idx}'] = self.grads[f'dZ{idx+1}'] @ self.cache[f'W{idx+1}'].T *\
                                        self.layers[idx].activation_fn(self.cache[f'Z{idx}'], derivative=True)


            self.grads[f'dW{idx}'] = 1 / m * self.layers[idx].input.T @ self.grads[f'dZ{idx}'] + reg*self.layers[idx].W
            self.grads[f'db{idx}'] = 1 / m * np.sum(self.grads[f'dZ{idx}'], axis=0, keepdims=True)
            
            assert self.grads[f'dW{idx}'].shape == self.cache[f'W{idx}'].shape

    def GDoptimize(self, idx, epoch_num, steps, learning_rate=1e-3):
        
        self.layers[idx].W -= learning_rate * self.grads[f'dW{idx}']
        self.layers[idx].b -= learning_rate * self.grads[f'db{idx}']

    
    def SGDMoptimize(self, idx, epoch_num, steps, learning_rate=1e-3, mu=0.99):
        m = dict()
        for i in self.layers.keys():
            m[f'W{i}'] = 0
            m[f'b{i}'] = 0

        m[f'W{idx}'] = m[f'W{idx}'] * mu - learning_rate * self.grads[f'dW{idx}']
        m[f'b{idx}'] = m[f'b{idx}'] * mu - learning_rate * self.grads[f'db{idx}']

        self.layers[idx].W += m[f'W{idx}']
        self.layers[idx].b += m[f'b{idx}']

    def Nesterovoptimize(self, idx, epoch_num, steps, learning_rate=1e-3, mu=0.99):
        m = dict()
        for i in self.layers.keys():
            m[f'W{i}'] = 0
            m[f'b{i}'] = 0

        mW_prev =  np.array(m[f'W{idx}'], copy=True)
        mb_prev = np.array(m[f'b{idx}'], copy=True)

        m[f'W{idx}'] = m[f'W{idx}'] * mu - learning_rate * self.grads[f'dW{idx}']
        m[f'b{idx}'] = m[f'b{idx}'] * mu - learning_rate * self.grads[f'db{idx}']
    
        w_update = -mu * mW_prev + (1 + mu) * m[f'W{idx}']
        b_update = -mu * mb_prev + (1 + mu) * m[f'b{idx}']

        self.layers[idx].W += w_update
        self.layers[idx].b += b_update

    def RMSpropoptimize(self, idx, epoch_num, steps,learning_rate=1e-3,decay_rate=0.99, epsilon=1e-8):
        v = dict()
        for i in self.layers.keys():
            v[f'W{i}'] = 0
            v[f'b{i}'] = 0
        v[f'W{idx}'] = decay_rate * v[f'W{idx}'] + (1 - decay_rate) * self.grads[f'dW{idx}'] **2 
        v[f'b{idx}'] = decay_rate * v[f'b{idx}'] + (1 - decay_rate) * self.grads[f'db{idx}'] **2
            
        w_update = -learning_rate * self.grads[f'dW{idx}'] / (np.sqrt(v[f'W{idx}'] + epsilon))
        b_update = -learning_rate * self.grads[f'db{idx}'] / (np.sqrt(v[f'b{idx}']+ epsilon))

        self.layers[idx].W += w_update
        self.layers[idx].b += b_update

    def Adamoptimize(self, idx, epoch_num, steps,learning_rate=1e-3, beta1=0.99, beta2=0.999, epsilon=1e-8): 
        m = dict()
        v = dict()

        for i in self.layers.keys():
            m[f'W{i}'] = 0
            m[f'b{i}'] = 0
            v[f'W{i}'] = 0
            v[f'b{i}'] = 0

        dW = self.grads[f'dW{idx}']
        db = self.grads[f'db{idx}']

        # weights
        m[f'W{idx}'] = beta1 * m[f'W{idx}'] + (1 - beta1) * dW
        v[f'W{idx}'] = beta2 * v[f'W{idx}'] + (1 - beta2) * dW ** 2 
        
        # biases
        m[f'b{idx}'] = beta1 * m[f'b{idx}'] + (1 - beta1) * db
        v[f'b{idx}'] = beta2 * v[f'b{idx}'] + (1 - beta2) * db ** 2 

        # take timestep into account
        mt_w  = m[f'W{idx}'] / (1 - beta1 ** steps)
        vt_w = v[f'W{idx}'] / (1 - beta2 ** steps)

        mt_b  = m[f'b{idx}'] / (1 - beta1 ** steps)
        vt_b = v[f'b{idx}'] / (1 - beta2 ** steps)

        w_update = - learning_rate * mt_w / (np.sqrt(vt_w) + epsilon)
        b_update = - learning_rate * mt_b / (np.sqrt(vt_b) + epsilon)

        self.layers[idx].W += w_update
        self.layers[idx].b += b_update

    def Nadamoptimize(self, idx, epoch_num, steps,learning_rate=1e-3, beta1=0.99, beta2=0.999, epsilon=1e-8): 
        m = dict()
        v = dict()

        for i in self.layers.keys():
            m[f'W{i}'] = 0
            m[f'b{i}'] = 0
            v[f'W{i}'] = 0
            v[f'b{i}'] = 0
        dW = self.grads[f'dW{idx}']
        db = self.grads[f'db{idx}']
            # weights
        m[f'W{idx}'] = beta1 * m[f'W{idx}'] + (1 - beta1) * dW
        v[f'W{idx}'] = beta2 * v[f'W{idx}'] + (1 - beta2) * dW ** 2 
            
            # biases
        m[f'b{idx}'] = beta1 * m[f'b{idx}'] + (1 - beta1) * db
        v[f'b{idx}'] = beta2 * v[f'b{idx}'] + (1 - beta2) * db ** 2 

            # take timestep into account
        mt_w  = m[f'W{idx}'] / (1 - beta1 ** steps)
        vt_w = v[f'W{idx}'] / (1 - beta2 ** steps)

        mt_b  = m[f'b{idx}'] / (1 - beta1 ** steps)
        vt_b = v[f'b{idx}'] / (1 - beta2 ** steps)

        w_update = - learning_rate / (np.sqrt(vt_w) + epsilon) * (beta1 * mt_w + (1 - beta1) *  dW / (1 - beta1 ** steps))
        b_update = - learning_rate / (np.sqrt(vt_b) + epsilon) * (beta1 * mt_b + (1 - beta1) *  db / (1 - beta1 ** steps))

        self.layers[idx].W += w_update
        self.layers[idx].b += b_update
            
    def fit(self, x_train, y_train,batch_size=32,epochs=500, learning_rate=1e-3, optimizer="GD",val_split=0.1,init_type="Xavier",loss_type="CrossEntropy",reg=0):
        '''Training cycle of the model object'''
        losses = []
        train_accs = []
        val_accs = []
        help=Helper()
        
        self.epochs = epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.init_type=init_type
        self.reg=reg
        self.loss_type=loss_type

        x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=val_split,stratify=y_train,random_state=42)

        for i in range(1, self.epochs+1):
            print(f'Epoch {i}')
            batches = help.create_batches(x_train, y_train, batch_size)
            epoch_loss = []
            steps = 0
            
            for x, y in batches:
                steps += 1
                preds = self.forward(x,self.init_type)
                #loss = help.compute_loss(y, preds,self.layers,self.loss_type,self.reg)
                #epoch_loss.append(loss)

                # Backward propagation - calculation of gradients 
                self.backward(y,self.loss_type,self.reg)
                
                # update weights and biases of each layer
                for idx in self.layers.keys():
                    if self.optimizer =="GD":
                        self.GDoptimize(idx, i, steps,learning_rate=self.learning_rate)
                    elif self.optimizer=="SGDM":
                        self.SGDMoptimize(idx, i, steps,learning_rate=self.learning_rate)
                    elif self.optimizer=="Nesterov":
                        self.Nesterovoptimize(idx, i, steps,learning_rate=self.learning_rate)
                    elif self.optimizer=="RMSprop":
                        self.RMSpropoptimize(idx, i, steps,learning_rate=self.learning_rate)
                    elif self.optimizer=="Adam":
                        self.Adamoptimize(idx, i, steps,learning_rate=self.learning_rate)
                    elif self.optimizer=="Nadam":
                        self.Nadamoptimize(idx, i, steps,learning_rate=self.learning_rate)
                
            #loss = sum(epoch_loss) / len(epoch_loss)
            #losses.append(loss)

            # Predict with network on x_train
            train_preds = self.forward(x_train)
            train_loss = help.compute_loss(y, preds,self.layers,self.loss_type,self.reg)
            train_acc=help.accuracy(train_preds,y_train)
            train_accs.append(train_acc)
            
            # Predict with network on x_val

            val_preds = self.forward(x_val)
            val_acc=help.accuracy(val_preds,y_val)
            val_accs.append(val_acc)
            val_loss = help.compute_loss(y_val, val_preds,self.layers,self.loss_type,self.reg)

            print(f'Train Loss:{train_loss} Train Acc: {train_acc} Val Acc: {val_acc} Val Loss: {val_loss}')   
            wandb.log(
        {"Train/Loss": train_loss, "Train/Accuracy": train_acc, "Val/Accuracy": val_acc, "Val/Loss":val_loss,"Epoch":i})
                
        self.history = {'train_loss': losses, 'train_acc': train_accs, 'val_acc': val_accs}
        
        


    def predict(self,x):
        preds=self.forward(x)
        return preds