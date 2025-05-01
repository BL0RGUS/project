# %%
import torch
import torchvision.datasets
import torchvision.transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64) 
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
  

        self.fc1 = nn.Linear(128*4*4,400, bias =  False)
        self.fc2 = nn.Linear(400, 10, bias = False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu (x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x


        
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = AlexNet().to(device)
network_state_dict = torch.load('AlexNetSmallfc.pth')
model.load_state_dict(network_state_dict)

print(model.fc1.weight.t().shape)
print(model.fc2.weight.t().shape)

model.eval()

# %%
test_dataset = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10(root='imgs/', train=False,
                               download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
  batch_size=32, shuffle=True)

values = [[], [], [], [], [], [], []]

for img, label in test_dataset:
    img, label = img.to(device), label.to(device)
    x = model.bn1(model.conv1(img))
    
    v = x.reshape(-1)
    values[0].append(v.max())
    values[0].append(v.min())
    
    x = F.relu(x)
    x = model.conv2(x)
    x = model.bn2(x)
    
    v = x.reshape(-1)
    values[1].append(v.max())
    values[1].append(v.min())
    
    x = F.relu(x)
    x = model.conv3(x)
    x = model.bn3(x)
    
    v = x.reshape(-1)
    values[2].append(v.max())
    values[2].append(v.min())
    
    x = F.relu(x)
    x = model.conv4(x)
    x = model.bn4(x)
    
    v = x.reshape(-1)
    values[3].append(v.max())
    values[3].append(v.min())
    
    x = F.relu(x)
    x = model.conv5(x)
    x = model.bn5(x)
    
    v = x.reshape(-1)
    values[4].append(v.max())
    values[4].append(v.min())
    
    x = F.relu(x)    
    x = x.view(x.size(0), -1)
    x = model.fc1(x)
    
    v = x.reshape(-1)
    values[5].append(v.max())
    values[5].append(v.min())
    
    
for i in range(6):
    print("Min: {}\nMax: {}\nSuggested Delta: {}".format(min(values[i]), max(values[i]), 1 / (max(abs(min(values[i])), abs(max(values[i]))))))
