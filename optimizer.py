import numpy as np

class Optimizer:
    def __init__(self, learning_rate=None, name=None):
        self.learning_rate = learning_rate
        self.name = name

    def config(self, layers):
        # sets up empty cache dictionaries 
        pass

    def optimize(self, idx, layers: list, grads: dict, *args):
        '''# Args: Takes in idx of the layer, list of the layers and the gradients as a dictionary 
            Performs updates in the list of layers passed into it'''
        pass 


class SGDM(Optimizer):

    def __init__(self, learning_rate=1e-2, mu_init=0.5, max_mu=0.99, demon=False, beta_init=0.9, **kwargs):
            super().__init__(**kwargs)
            self.mu_init = mu_init
            self.max_mu = max_mu
            self.demon = demon
            if self.demon:
                self.beta_init = beta_init
            self.m = dict()

    def config(self, layers):
        for i in layers.keys():
            self.m[f'W{i}'] = 0
            self.m[f'b{i}'] = 0

    def optimize(self, idx, layers, grads, epoch_num, steps):
        # increase mu by a factor of 1.2 every epoch until max_mu is reached (only applicable for momentum and nesterov momentum)
        mu = min(self.mu_init * 1.2 ** (epoch_num - 1), self.max_mu)

        if self.demon:
            p_t = 1 - epoch_num / self.epochs 
            mu = self.beta_init * p_t / ((1 - self.beta_init) + self.beta_init * p_t) 

        self.m[f'W{idx}'] = self.m[f'W{idx}'] * mu - self.learning_rate * grads[f'dW{idx}']
        self.m[f'b{idx}'] = self.m[f'b{idx}'] * mu - self.learning_rate * grads[f'db{idx}']

        layers[idx].W += self.m[f'W{idx}']
        layers[idx].b += self.m[f'b{idx}']



class Nesterov(SGDM):
    def __init__(self, learning_rate, **kwargs):
        self.learning_rate = learning_rate
        super().__init__(**kwargs)


    def optimize(self, idx, layers, grads, epoch_num, steps):
        # increase mu by a factor of 1.2 every epoch until max_mu is reached (only applicable for momentum and nesterov momentum)
        mu = min(self.mu_init * 1.2 ** (epoch_num - 1), self.max_mu)
        if self.demon:
            p_t = 1 - epoch_num / self.epochs 
            mu = self.beta_init * p_t / ((1 - self.beta_init) + self.beta_init * p_t) 

        mW_prev =  np.array(self.m[f'W{idx}'], copy=True)
        mb_prev = np.array(self.m[f'b{idx}'], copy=True)

        self.m[f'W{idx}'] = self.m[f'W{idx}'] * mu - self.learning_rate * grads[f'dW{idx}']
        self.m[f'b{idx}'] = self.m[f'b{idx}'] * mu - self.learning_rate * grads[f'db{idx}']
    
        w_update = -mu * mW_prev + (1 + mu) * self.m[f'W{idx}']
        b_update = -mu * mb_prev + (1 + mu) * self.m[f'b{idx}']

        layers[idx].W += w_update
        layers[idx].b += b_update


class RMSprop(Optimizer):
    def __init__(self, decay_rate=0.9, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = dict()

    def config(self, layers):
        for i in layers.keys():
            self.cache[f'W{i}'] = 0
            self.cache[f'b{i}'] = 0

    def optimize(self, idx, layers, grads, epoch_num, steps):
        self.cache[f'W{idx}'] = self.decay_rate * self.cache[f'W{idx}'] + (1 - self.decay_rate) * grads[f'dW{idx}'] **2 
        self.cache[f'b{idx}'] = self.decay_rate * self.cache[f'b{idx}'] + (1 - self.decay_rate) * grads[f'db{idx}'] **2
        
        w_update = - self.learning_rate * grads[f'dW{idx}'] / (np.sqrt(self.cache[f'W{idx}'] + self.epsilon))
        b_update = - self.learning_rate * grads[f'db{idx}'] / (np.sqrt(self.cache[f'b{idx}']+ self.epsilon))

        layers[idx].W += w_update
        layers[idx].b += b_update


class Adam(Optimizer):
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, 
                 weight_decay=False, gamma_init=1e-5, decay_rate=0.8, demon=False, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2 
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        if self.weight_decay:
            self.gamma_init = gamma_init
            self.decay_rate = decay_rate
        self.demon = demon
        self.m = dict()  # first moment estimate 
        self.v = dict()  # second raw moment estimate 

    def config(self, layers):
        for i in layers.keys():
            self.m[f'W{i}'] = 0
            self.m[f'b{i}'] = 0
            self.v[f'W{i}'] = 0
            self.v[f'b{i}'] = 0

    def optimize(self, idx, layers, grads, epoch_num, steps): 
        dW = grads[f'dW{idx}']
        db = grads[f'db{idx}']
        if self.demon:
            p_t = 1 - epoch_num / self.epochs
            beta1 = self.beta1 * (p_t / (1 - self.beta1 + self.beta1 * p_t))
        else:
            beta1 = self.beta1

        # weights
        self.m[f'W{idx}'] = beta1 * self.m[f'W{idx}'] + (1 - beta1) * dW
        self.v[f'W{idx}'] = self.beta2 * self.v[f'W{idx}'] + (1 - self.beta2) * dW ** 2 
        
        # biases
        self.m[f'b{idx}'] = beta1 * self.m[f'b{idx}'] + (1 - beta1) * db
        self.v[f'b{idx}'] = self.beta2 * self.v[f'b{idx}'] + (1 - self.beta2) * db ** 2 

        # take timestep into account
        mt_w  = self.m[f'W{idx}'] / (1 - beta1 ** steps)
        vt_w = self.v[f'W{idx}'] / (1 - self.beta2 ** steps)

        mt_b  = self.m[f'b{idx}'] / (1 - beta1 ** steps)
        vt_b = self.v[f'b{idx}'] / (1 - self.beta2 ** steps)

        w_update = - self.learning_rate * mt_w / (np.sqrt(vt_w) + self.epsilon)
        b_update = - self.learning_rate * mt_b / (np.sqrt(vt_b) + self.epsilon)
        
        if self.weight_decay:
            gamma = self.gamma_init * self.decay_rate ** int(epoch_num / 5) 
            w_update = - self.learning_rate * mt_w / ((np.sqrt(vt_w) + self.epsilon) + gamma * layers[idx].W) 
            b_update = - self.learning_rate * mt_b / ((np.sqrt(vt_b) + self.epsilon) + gamma * layers[idx].b)

        layers[idx].W += w_update
        layers[idx].b += b_update

class Nadam(Adam):
    def __init__(self, learning_rate, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate

    def optimize(self, idx, layers, grads, epoch_num, steps): 
        dW = grads[f'dW{idx}']
        db = grads[f'db{idx}']

        if self.demon:
            p_t = 1 - epoch_num / self.epochs
            beta1 = self.beta1 * (p_t / (1 - self.beta1 + self.beta1 * p_t))
        else:
            beta1 = self.beta1

        # weights
        self.m[f'W{idx}'] = beta1 * self.m[f'W{idx}'] + (1 - beta1) * dW
        self.v[f'W{idx}'] = self.beta2 * self.v[f'W{idx}'] + (1 - self.beta2) * dW ** 2 
        
        # biases
        self.m[f'b{idx}'] = beta1 * self.m[f'b{idx}'] + (1 - beta1) * db
        self.v[f'b{idx}'] = self.beta2 * self.v[f'b{idx}'] + (1 - self.beta2) * db ** 2 

        # take timestep into account
        mt_w  = self.m[f'W{idx}'] / (1 - beta1 ** steps)
        vt_w = self.v[f'W{idx}'] / (1 - self.beta2 ** steps)

        mt_b  = self.m[f'b{idx}'] / (1 - beta1 ** steps)
        vt_b = self.v[f'b{idx}'] / (1 - self.beta2 ** steps)

        if self.weight_decay:
            gamma = self.gamma_init * self.decay_rate ** int(epoch_num / 5) 
            w_update = - self.learning_rate / (np.sqrt(vt_w) + self.epsilon + gamma * layers[idx].W) * (beta1 * mt_w + (1 - beta1) *  dW / (1 - beta1 ** steps))
            b_update = - self.learning_rate / (np.sqrt(vt_b) + self.epsilon + gamma * layers[idx].b) * (beta1 * mt_b + (1 - beta1) *  db / (1 - beta1 ** steps))
        else:
            w_update = - self.learning_rate / (np.sqrt(vt_w) + self.epsilon) * (beta1 * mt_w + (1 - beta1) *  dW / (1 - beta1 ** steps))
            b_update = - self.learning_rate / (np.sqrt(vt_b) + self.epsilon) * (beta1 * mt_b + (1 - beta1) *  db / (1 - beta1 ** steps))

        layers[idx].W += w_update
        layers[idx].b += b_update