import torch
import torchvision.datasets
import torchvision.transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

n_epochs = 50
batch_size_train = 200
batch_size_test = 200
learning_rate = 0.001
log_interval = 100

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10(root='imgs/', train=True,
                               download=True, transform=torchvision.transforms.Compose([
                                torchvision.transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
                                torchvision.transforms.RandomRotation(10),  # Rotates the image to a specified angel
                                torchvision.transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Performs actions like zooms, change shear angles.
                                torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
                               torchvision.transforms.ToTensor()
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10(root='imgs/', train=False,
                               download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
  batch_size=batch_size_test, shuffle=True)


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
  

        self.fc1 = nn.Linear(128*4*4,1000, bias =  False)
        self.fc2 = nn.Linear(1000, 100, bias = False) 
        self.fc3 = nn.Linear(100, 10, bias = False)
        
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
        x = F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x

        
        
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
network =AlexNet().to(device)
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

network.eval()
lossfn = nn.CrossEntropyLoss()


train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch, device):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    output = network(data)
    loss = lossfn(output, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), 'AlexNetSmallfc.pth')
      torch.save(optimizer.state_dict(), 'optimizer.pth')
      
      
def test(device):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = network(data)
      test_loss += lossfn(output, target).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  

for epoch in range(1, n_epochs + 1):
  train(epoch, device)
  test(device)

network.eval()

with torch.no_grad():
  torch.save(network.state_dict(), 'AlexNet.pth')