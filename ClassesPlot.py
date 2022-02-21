! pip install wandb
! wandb login
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import wandb
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_type = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] 

images=[]
labels=[]
i=0
while(len(labels)<10):
  if y_train[i] not in labels:
    labels.append(y_train[i])
    images.append(x_train[i])
    
  i+=1
  
  
wandb.init(project="Assignment_1_random_randomSeed_SE", entity="cs21m007_cs21m013")

num=10
num_row = 2
num_col = 5# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('Label: {}'.format(labels[i]))
plt.tight_layout()
wandb.log({'Clases':plt})