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
  torchvision.datasets.CIFAR10(root='./data', train=True,
                               download=True, transform=torchvision.transforms.Compose([
                                torchvision.transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
                                torchvision.transforms.RandomRotation(10),  # Rotates the image to a specified angel
                                torchvision.transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Performs actions like zooms, change shear angles.
                                torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ])),
  batch_size=batch_size, shuffle=True)

test_dataloader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ])),
  batch_size=1, shuffle=True)

x_train = np.zeros((int(len(train_dataloader.dataset)/10), 3, 32, 32))
y_train = np.zeros((int(len(train_dataloader.dataset)/10), 1))
idx = 0

for data, target in train_dataloader:
    target_np = target.cpu().numpy()
    for idx_batch, im in enumerate(data.numpy()):
        x_train[idx] = np.expand_dims(im, axis=0)
        y_train[idx] = target_np[idx_batch]
        idx += 1
        if idx >= len(train_dataloader.dataset)/10:
            break
    if idx >= len(train_dataloader.dataset)/10:
        break


x_test = np.zeros((len(test_dataloader.dataset), 3, 32, 32))
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
class AlexNet(nn.Module):
 
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
 
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

def train_one_epoch(net, optimizer, train_loader, device):
    # Cross Entropy loss for classification when not using a softmax layer in the network
    loss = nn.CrossEntropyLoss()
    net.train()
    avg_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = net(data)
        loss_net = loss(output, target.long())
        optimizer.zero_grad()
        loss_net.backward()
        optimizer.step()
        avg_loss += loss_net.item()

    return avg_loss / len(train_loader)


def test_torch(net, test_loader, device):
    """Test the network: measure accuracy on the test set."""

    # Freeze normalization layers
    net.eval()
    test_acc = 0
    # Iterate over the batches
    idx = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # Accumulate the ground truth labels

        # Run forward and get the predicted class id
        output = net(data)
        test_acc += (output.max(1)[1] == target).sum().item()

        idx += target.shape[0]

    # Print out the accuracy as a percentage
    print(
        f"Test accuracy for fp32 weights and activations: "
        f"{test_acc/ len(test_loader) * 100:.2f}%"
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
N_EPOCHS = 35

# Train the network with Adam, output the test set accuracy every epoch

PATH = './cifar_net.pth'
device = "cuda"
net = AlexNet(10)
if(training):
    losses_bits = []
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    for _ in range(N_EPOCHS):
        print(_)
        losses_bits.append(train_one_epoch(net, optimizer, train_dataloader, device))
        test_torch(net, test_dataloader, device)
    net.eval()
    torch.save(net.state_dict(), PATH)
    
else:
    net.load_state_dict(torch.load(PATH, weights_only=True, map_location="cpu"))
    
    #test_torch(net, test_dataloader)
    
    n_bits = {
    "model_inputs": 8,
    "op_inputs": 6,
    "op_weights": 6,
    "model_outputs": 9
    }

    
    rounding_bits =  {"n_bits": 6, "method": "APPROXIMATE"}
    use_gpu_if_available = True
    device = "cuda" if use_gpu_if_available and check_gpu_available() else "cpu"
    
    print(device)

    q_module = compile_torch_model(net, x_train, n_bits=n_bits, rounding_threshold_bits=rounding_bits, p_error=0.1, device=device)
    
    #start_time = time.time()
    #accs = test_with_concrete(
     #   q_module,
     #   test_dataloader,
     #   use_sim=True,
    #)
    #sim_time = time.time() - start_time

    #print(f"Simulated FHE execution for {n_bits} bit network accuracy: {100*accs:.2f}%")   
    # Generate keys first
    t = time.time()
    q_module.fhe_circuit.keygen()
    print(f"Keygen time: {time.time()-t:.2f}s")

    print(q_module.fhe_circuit.statistics)
    
    # Run inference in FHE on a single encrypted example
    
    test_data_length = 1 
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

    print(time_per_inference)
    print(accuracy_percentage)