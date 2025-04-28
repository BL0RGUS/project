# %%
import torch
import torchvision.datasets
import torchvision.transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


# %%
n_epochs = 30
batch_size_train = 500
batch_size_test = 500
learning_rate = 0.001
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


# %%

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('imgs/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('imgs/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
  batch_size=batch_size_test, shuffle=True)

for data, label in train_loader:
  print(data.shape)
  break


# %%
class CryptoNet(nn.Module):
    def __init__(self):
        super(CryptoNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(1960, 10, bias =  False)
        self.fc2 = nn.Linear(10, 10, bias = False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.square(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.square(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.square(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x

        
network = CryptoNet()
optimizer = optim.Adam(network.parameters(), lr = 0.00075)
lossfn = nn.CrossEntropyLoss()

# %%


train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
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
      torch.save(network.state_dict(), 'model_mnist.pth')
      torch.save(optimizer.state_dict(), 'optimizer_mnist.pth')
      
      
def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += lossfn(output, target).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

# %%
test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()

network.eval()

with torch.no_grad():
  torch.save(network.state_dict(), 'model_mnist.pth')

# %%
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig


