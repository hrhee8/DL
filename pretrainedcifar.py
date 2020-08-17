#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.transforms as transforms

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
# Loading the data

def resnet18(pretrained=True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],model_dir='~/scratch/'))
    return model

model = resnet18(pretrained=True)

# If you just need to fine-tune the last layer, comment out the code below.
# for param in model.parameters():
#     param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 100)

# We provide the code for loading CIFAR100 data
num_epochs = 50
batch_size = 128
# blocks = [2,4,4,2]
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
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
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
        print(test_accuracy)
        if test_accuracy > 0.7:
            break

