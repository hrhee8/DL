#!/usr/bin/env python
# coding: utf-8

# In[9]:


#setup for implementing gpu brought from given file
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import h5py
import time


# In[3]:


#data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),   
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# In[5]:


train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size= 100,
                                          shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=100,
                                         shuffle=False, num_workers=2)


# In[19]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(3,64,4,1,2),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,4,1,2),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.5)
        )
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(64,128,4,1,2),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,128,4,1,2),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.5)
        )
        self.convlayer3 = nn.Sequential(
            nn.Conv2d(128,256,4,1,2),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,256,4,1,2),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.5)
        )
        self.fulllayer = nn.Sequential(
            nn.Linear(256*5*5,500),
            nn.ReLU(inplace = True),
            nn.Linear(500,10)
        )
        
    def forward(self,x):
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = x.view(-1,256*5*5)
        x = self.fulllayer(x)
        return x


# In[20]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones = [60,120,160],gamma=0.1, last_epoch=-1)


# In[ ]:


epoch = 9
time1 = time.time()

for epoch in range(epoch):
    scheduler.step()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print ("Epoch", epoch+1, "%s minibatches"%i, "loss: %.4f"%(running_loss/1000.))
            running_loss = 0.0
   
print('Complete')
time2 = time.time()
print(time2-time1)


# In[22]:


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Test accuracy:", (100 * correct / total))


# In[ ]:




