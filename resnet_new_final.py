#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import time
import torch.nn as nn




# We provide the code for loading CIFAR100 data
num_epochs = 50
batch_size = 128
blocks = [2,4,4,2]
num_classes=100
# torch.manual_seed(0)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR100(root='~/scratch/', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR100(root='~/scratch/', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
#         self.convsub = nn.Sequential()
#         if self.downsample == True:
#             self.convsub = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(out_channels))
        
#         self.relu = nn.ReLU(inplace = True)
#         self.downsample = downsample
#         self.stride = stride

#         self.shortcut = nn.Sequential()
#         if stride!=1 or in_channels!=out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels))


        self.convsub = nn.Sequential()
        if stride!=1 or in_channels!=out_channels:
            self.convsub = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.conv(x)
        if self.downsample == True:
            out += self.convsub(x)
        return nn.ReLU(inplace=True)(out)
#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.conv(x) + self.shortcut(x))

class ResNet(nn.Module):
    def __init__(self, basic_block, blocks, num_classes):
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2))
        
        self.conv2_x = self.layer(basic_block, 32, 32, blocks[0],1)
        self.conv3_x = self.layer(basic_block, 32, 64, blocks[1], 2)
        self.conv4_x = self.layer(basic_block, 64, 128, blocks[2], 2)
        self.conv5_x = self.layer(basic_block, 128, 256, blocks[3], 2)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.fc = nn.Linear(256, num_classes)

    def layer(self, block, in_channels, out_channels, blocks, stride):
        if stride >1:
            downsample = True
        else: downsample = False
        layers = []
        layers.append(block(in_channels, out_channels, stride=stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
#         strides = [stride] + [1] *(blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(in_channels, out_channels, stride))
#             in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
#         print(x.shape)
        output = self.conv1(x)
#         print(output.shape)
        output = self.conv2_x(output)
#         print(output.shape)
        output = self.conv3_x(output)
#         print("here1")
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.pool(output)
#         print("here")
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output 


# YOUR CODE GOES HERE
# Change to your ResNet
model = ResNet(BasicBlock, blocks, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
for epoch in range(num_epochs):
    # Train the model
    model.train()
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
        X_train_batch,Y_train_batch = X_train_batch.to(device),Y_train_batch.to(device)
        outputs = model(X_train_batch)
        loss = criterion(outputs, Y_train_batch)
        optimizer.zero_grad()
        loss.backward()
        if epoch > 16:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    state = optimizer.state[p]
                    if state["step"] >= 1024:
                        state["step"] = 1000
        optimizer.step()
       
    # Test the model
    with torch.no_grad():
        model.eval()
        correct = []
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
            X_test_batch, Y_test_batch= X_test_batch.to(device),Y_test_batch.to(device)
            output = model(X_test_batch)
            predict = output.max(1)[1]
            correct.append(float(predict.eq(Y_test_batch).sum()) / float(batch_size))
        test_accuracy = sum(correct) / len(correct)
        print(epoch, test_accuracy)
        if test_accuracy > 0.6:
            break


# In[ ]:




