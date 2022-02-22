import numpy as np
class Helper:
    def accuracy(self,y,y_hat):
        c = np.argmax(y_hat, axis=1) == np.argmax(y, axis=1)
        acc = list(c).count(True) / len(c) * 100
        return acc

    def compute_loss(self,Y, Y_hat,layers,loss_type="CrossEntropy",reg=0):
        if loss_type=="CrossEntropy":
            m = Y.shape[0]
            L = -1./m * np.sum(Y * np.log(Y_hat+0.0000000001))
        elif loss_type=="SquaredError":
            L = np.mean((Y- Y_hat)**2)

        if reg!=0:
            reg_error = 0.0                                                                       
            for idx in layers.keys() :
              reg_error += (reg/2)*(np.sum(np.square(layers[idx].W))) 
            L = L + reg_error

        return L

    
    def create_batches(self,x, y, batch_size):
        m = x.shape[0]
        num_batches = m / batch_size
        batches = []
        for i in range(int(num_batches+1)):
            batch_x = x[i*batch_size:(i+1)*batch_size]
            batch_y = y[i*batch_size:(i+1)*batch_size]
            batches.append((batch_x, batch_y))
        
        if m % batch_size == 0:
            batches.pop(-1)

        return batches
    