#!/usr/bin/env python
# coding: utf-8

# In[1]:


### I have manipulated the example code that is given in the lecture note to show the resultant of the homework

import numpy as np
import h5py
import time
import copy
from random import randint
#load MNIST data
MNIST_data = h5py.File('MNISTdata_1.hdf5', 'r')
print(list(MNIST_data.keys()))


# In[2]:


x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0])) 
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()


# In[3]:


####################################################################################
#Implementation of stochastic gradient descent algorithm

d = 28
kx,ky = 2,2
c = 3
#output
output = 10


#Randomly create array with certain dimension
model = {}
model['W'] = np.random.randn(output,d-ky+1,d-kx+1,c) * np.sqrt(1 / (output*(d-ky+1)*(d-kx+1)*c))
#print(model['W'].shape)
model['bk'] = np.zeros((output,1))
model['K'] = np.random.randn(kx,ky,c) * np.sqrt(1 / (kx * ky))

#make a copy for later to implement
model_grads = copy.deepcopy(model)


            
#convolution m,n,p p is C


def convZ(X, K):
    k_y,k_x = K.shape[0],K.shape[1]
    conv = np.zeros(((d-k_y+1),(d-k_x+1)))
    for i in range(d-k_y+1):
        for j in range(d-k_x+1):
                conv[i][j] = np.sum(X[i:i+k_y-1, j:j+k_x-1]*K)
    conv = np.array(conv)
    return conv

#define function for convolution
def relu(X):
    return np.maximum(0,X)

def backrelu(x):
    return 1. * (x > 0)


def softmax(z):
    ZZ = np.exp(z)/np.sum(np.exp(z)) 
    return ZZ

def forward(x,y, model):
    x = x.reshape((784,1))
    x = x.reshape((d,d))
    result = np.zeros(((d-ky+1),(d-kx+1),c))
    H = np.zeros(((d-ky+1),(d-kx+1),c))
    for p in range(c):
        result[:,:,p] = convZ(x,model['K'][:,:,p])
    model['Z'] = result
    #print(model['Z'].shape,"hi")
    #print(model['W'].shape)
    H[:,:,p] = relu(result[:,:,p])
    model['H'] = H
    #print(model['H'])
    #print(model['bk'].shape)
    #print(np.einsum('ijkl,jkl -> i',model['W'],model['H']).reshape(output,1))
    #print(model['bk'])
    Uk = (np.einsum('ijkl,jkl -> i',model['W'],model['H']).reshape(output,1) + model['bk'])
    #print(Uk.shape)
    f = softmax(Uk)
    #print(f.shape)
    return f


def sigmoidprime(z):
    sprime = 1/(1+np.exp(-z))
    return sprime * (1-sprime)



def backward(x,y,f, model, model_grads): 
    dpdU = -1.0*f
    dpdU[y] = dpdU[y] + 1.0
    dpdU = dpdU*-1
    dpdbk = dpdU
    dpdW = np.zeros(model['W'].shape)
    delta = np.zeros((d-ky+1,d-kx+1,c))

    #deltaijp = np.dot(dpdU,model['W'].reshape(model['W'].shape[0],np.prod(model['H'].shape))).reshape(model['H'].shape)
    #print(deltaijp)
    #deltaijp = np.einsum('ij,ijkl -> jkl',dpdU,model['W'])
    deltaijp = np.dot(model['W'].reshape(output,np.prod(model['H'].shape)).T,dpdU)
    deltaijp = deltaijp.reshape((d-ky+1,d-kx+1,c))
    delta = deltaijp
    #print(delta.shape)
    #print(deltaijp.shape)
    #delta = deltaijp.reshape((d-ky+1,d-kx+1,c))
    #print(delta)
    
    for i in range(K):
        dpdW[i] = dpdU[i] * model['H'][i]
    deltaK = np.zeros(model['K'].shape)
    Z = model['Z']
    H = model['H']
    
    model_grads['W'] = np.dot(dpdU.reshape(output,1), H.reshape(np.prod(H.shape),1).T).reshape(model_grads['W'].shape)
    #print(Z)
    
 
    
    for p in range(c):
        print(Z[:,:,p].shape)
        print(delta[:,:,p].shape)
        dk = np.multiply(backrelu(Z[:,:,p]),delta[:,:,p])
        print(dk)
        print(convZ(x,dk))
        deltaK[:,:,p] = convZ(x,dk)
    model_grads['W'] = dpdW
    model_grads['K'] = deltaK
    model_grads['bk'] = dpdbk
    return model_grads



# In[5]:


#### train the model

import time
time1 = time.time() 
LR = .01
num_epochs = 20
for epochs in range(num_epochs):
    #Learning rate schedule
    if (epochs > 5): 
        LR = 0.001
    if (epochs > 10): 
        LR = 0.0001
    if (epochs > 15): 
        LR = 0.00001
        
    total_correct = 0   
    for n in range(len(x_train)):
        n_random = randint(0,len(x_train)-1 ) 
        y = y_train[n_random]
        x = x_train[n_random][:]
        f = forward(x, y, model)
        prediction = np.argmax(f)
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x,y,f, model, model_grads)
        model['W'] = model['W'] + LR*model_grads['W']
        model['K'] = model['K'] + LR*model_grads['K']
        model['bk'] = model['bk'] + LR*model_grads['bk']
    print("training in process..", "(",epochs+1,"/",num_epochs,")")
    print("Accuracy is",(total_correct/np.float(len(x_train)))*100,"%") 
    
time2 = time.time()
print("Processing time was", time2-time1)
###################################################### #test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    f = forward(x, y, model)
    prediction = np.argmax(f) 
    if (prediction == y):
        total_correct += 1 
        
print("Test Result: ",total_correct/np.float(len(x_test)), "%")


# In[ ]:





# In[ ]:




