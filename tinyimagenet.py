#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch.nn as nn
import torch.optim as optim
# num_epochs = 1
# batch_size = 128
blocks = [2,4,4,2]
num_classes=200

num_epochs = 40
batch_size = 128

transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
])

def create_val_folder(val_dir):
    """
    This method is responsible for separating validation images into separate sub folders
    """
    path = os.path.join(val_dir, 'images')  # path where validation data is present now
    filename = os.path.join(val_dir, 'val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()
    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)
        if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))
    return

# Your own directory to the train folder of tiyimagenet
train_dir = 'tiny-imagenet-200/train/'
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
# To check the index for each classes
# print(train_dataset.class_to_idx)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
# Your own directory to the validation folder of tiyimagenet
val_dir = 'tiny-imagenet-200/val/'


if 'val_' in os.listdir(val_dir+'images/')[0]:
    create_val_folder(val_dir)
    val_dir = val_dir+'images/'
else:
    val_dir = val_dir+'images/'


val_dataset = datasets.ImageFolder(val_dir, transform=transforms.ToTensor())
# To check the index for each classes
# print(val_dataset.class_to_idx)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)


# YOUR CODE GOES HERE
# Change to your ResNet
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

class ResNet(nn.Module):
    def __init__(self, basic_block, blocks, num_classes):
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1))
        
        self.conv2_x = self.layer(basic_block, 32, 32, blocks[0],1)
        self.conv3_x = self.layer(basic_block, 32, 64, blocks[1], 2)
        self.conv4_x = self.layer(basic_block, 64, 128, blocks[2], 2)
        self.conv5_x = self.layer(basic_block, 128, 256, blocks[3], 2)
        self.pool = nn.MaxPool2d(kernel_size=8, stride=8)
        self.fc = nn.Linear(256, num_classes)

    def layer(self, block, in_channels, out_channels, blocks, stride):
        if stride >1:
            downsample = True
        else: downsample = False
        layers = []
        layers.append(block(in_channels, out_channels, stride=stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
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




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNet(BasicBlock, blocks, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0, momentum=0.8)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.2)
for epoch in range(num_epochs):
    # Train the model
    model.train()
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(train_loader):
        X_train_batch,Y_train_batch = X_train_batch.to(device),Y_train_batch.to(device)
        outputs = model(X_train_batch)
        loss = criterion(outputs, Y_train_batch)
        optimizer.zero_grad()
        loss.backward()
        #if epoch > 7:
         #   for group in optimizer.param_groups:
          #      for p in group["params"]:
           #         state = optimizer.state[p]
            #        if state["step"] >= 1024:
             #           state["step"] = 1000
        optimizer.step()
    scheduler.step()   
    # Test the model
    with torch.no_grad():
        model.eval()
        correct = []
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(val_loader):
            X_test_batch, Y_test_batch= X_test_batch.to(device),Y_test_batch.to(device)
            output = model(X_test_batch)
            predict = output.max(1)[1]
            correct.append(float(predict.eq(Y_test_batch).sum()) / float(batch_size))
        test_accuracy = sum(correct) / len(correct)
        print(epoch, test_accuracy)
        if test_accuracy > 0.5:
            break

