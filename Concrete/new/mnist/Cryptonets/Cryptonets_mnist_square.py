import time

import numpy as np
import torch
import torch.utils
from concrete.compiler import check_gpu_available

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from concrete.ml.torch.compile import compile_torch_model

import torchvision





batch_size = 200


train_dataloader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=True)

test_dataloader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=1, shuffle=True)

x_train = np.zeros((len(train_dataloader.dataset), 1, 28, 28))
y_train = np.zeros((len(train_dataloader.dataset), 1))
idx = 0

for data, target in train_dataloader:
    target_np = target.cpu().numpy()
    for idx_batch, im in enumerate(data.numpy()):
        x_train[idx] = np.expand_dims(im, axis=0)
        y_train[idx] = target_np[idx_batch]
        idx += 1


x_test = np.zeros((len(test_dataloader.dataset), 1, 28, 28))
y_test = np.zeros((len(test_dataloader.dataset), 1))
idx = 0

print(x_test.shape)
print(x_train.shape)
for data, target in test_dataloader:
    target_np = target.cpu().numpy()
    for idx_batch, im in enumerate(data.numpy()):
        x_test[idx] = np.expand_dims(im, axis=0)
        y_test[idx] = target_np[idx_batch]
        idx += 1
        
training = False

class TinyCNN(nn.Module):
    """A very small CNN to classify the sklearn digits data-set."""

    def __init__(self, n_classes) -> None:
        """Construct the CNN with a configurable number of classes."""
        super().__init__()

        # This network has a total complexity of 1216 MAC
        self.conv1 = nn.Conv2d(1, 5, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 10, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(490, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Run inference on the tiny CNN, apply the decision layer on the reshaped conv output."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.square(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.square(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = torch.square(x)
        x = self.fc2(x)
        if self.training:
            x = self.sigmoid(x)
        return x
    

def train_one_epoch(net, optimizer, train_loader):
    # Cross Entropy loss for classification when not using a softmax layer in the network
    loss = nn.CrossEntropyLoss()
    net.train()
    avg_loss = 0
    for data, target in train_loader:
        output = net(data)
        loss_net = loss(output, target.long())
        optimizer.zero_grad()
        loss_net.backward()
        optimizer.step()
        avg_loss += loss_net.item()

    return avg_loss / len(train_loader)

def test_torch(net, test_loader):
    """Test the network: measure accuracy on the test set."""

    # Freeze normalization layers
    net.eval()

    all_y_pred = np.zeros((len(test_loader)), dtype=np.int64)
    all_targets = np.zeros((len(test_loader)), dtype=np.int64)

    # Iterate over the batches
    idx = 0
    for data, target in test_loader:
        # Accumulate the ground truth labels
        endidx = idx + target.shape[0]
        all_targets[idx:endidx] = target.numpy()

        # Run forward and get the predicted class id
        output = net(data).argmax(1).detach().numpy()
        all_y_pred[idx:endidx] = output

        idx += target.shape[0]

    # Print out the accuracy as a percentage
    n_correct = np.sum(all_targets == all_y_pred)
    print(
        f"Test accuracy for fp32 weights and activations: "
        f"{n_correct / len(test_loader) * 100:.2f}%"
    )
    
def test_with_concrete(quantized_module, test_loader, use_sim):
    """Test a neural network that is quantized and compiled with Concrete ML."""

    # Casting the inputs into int64 is recommended
    all_y_pred = np.zeros((len(test_loader)), dtype=np.int64)
    all_targets = np.zeros((len(test_loader)), dtype=np.int64)

    # Iterate over the test batches and accumulate predictions and ground truth labels in a vector
    idx = 0
    for data, target in test_loader:
        data = data.numpy()
        target = target.numpy()

        fhe_mode = "simulate" if use_sim else "execute"

        # Quantize the inputs and cast to appropriate data type
        y_pred = quantized_module.forward(data, fhe=fhe_mode)

        endidx = idx + target.shape[0]

        # Accumulate the ground truth labels
        all_targets[idx:endidx] = target

        # Get the predicted class id and accumulate the predictions
        y_pred = np.argmax(y_pred, axis=1)
        all_y_pred[idx:endidx] = y_pred

        # Update the index
        idx += target.shape[0]

    # Compute and report results
    n_correct = np.sum(all_targets == all_y_pred)
    return n_correct / len(test_loader)

# Create the tiny CNN with 10 output classes
N_EPOCHS = 12

PATH = './mnist_square_net.pth'

# Train the network with Adam, output the test set accuracy every epoch
net = TinyCNN(10)
if(training):
    losses_bits = []
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for _ in range(N_EPOCHS):
        losses_bits.append(train_one_epoch(net, optimizer, train_dataloader))
    net.eval()
    
    torch.save(net.state_dict(), PATH)
    test_torch(net, test_dataloader)
else:
    net.load_state_dict(torch.load(PATH, weights_only=True))
    n_bits = 6

    use_gpu_if_available = True
    device = "cuda" if use_gpu_if_available and check_gpu_available() else "cpu"

    print(device)

    q_module = compile_torch_model(net, x_train, n_bits=n_bits, rounding_threshold_bits=6, p_error=0.1, device=device)
    
    start_time = time.time()
    accs = test_with_concrete(
        q_module,
        test_dataloader,
        use_sim=True,
    )
    sim_time = time.time() - start_time

    print(f"Simulated FHE execution for {n_bits} bit network accuracy: {100*accs:.2f}%")   

    # Generate keys first
    t = time.time()
    q_module.fhe_circuit.keygen()
    print(f"Keygen time: {time.time()-t:.2f}s")



    # Run inference in FHE on a single encrypted example
    test_data_length = 2 
    print(x_test[:test_data_length, :].shape)
    mini_test_dataset = TensorDataset(torch.Tensor(x_test[:test_data_length, :]), torch.Tensor(y_test[:test_data_length]))
    mini_test_dataloader = DataLoader(mini_test_dataset)

    t = time.time()
    accuracy_test = test_with_concrete(
        q_module,
        mini_test_dataloader,
        use_sim=False,
    )
    elapsed_time = time.time() - t
    time_per_inference = elapsed_time / len(mini_test_dataset)
    accuracy_percentage = 100 * accuracy_test

    print(
        f"Time per inference in FHE: {time_per_inference:.2f} "
        f"with {accuracy_percentage:.2f}% accuracy"
    )