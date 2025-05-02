import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import random

""" Abigail Gehlbach CSI-260 Advanced Python
    Tutorial from pytorch docs: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    CNN Done as a benchmark for the CIFAR10 image classification standard."""

# using torchvision transforms to normalize dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 4


#training and testing set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    """plot image"""
    img = img / 2 + 0.5     # de normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Net(nn.Module):
    """CNN model taken from tutorial"""
    def __init__(self):
        """initialize model"""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """forward pass"""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def time_inference_cnn(net, device, testloader, num_samples=100):
    """test classification for CNN num_samples number of times and returns average time per image and total time"""
    net.eval()

    images_list = []
    labels_list = []
    for images, labels in testloader:
        images_list.append(images)
        labels_list.append(labels)
    images_all = torch.cat(images_list)
    labels_all = torch.cat(labels_list)

    images = torch.cat(images_list)
    labels = torch.cat(labels_list)

    indices = random.sample(range(len(images_all)), num_samples)
    selected_images = images_all[indices].to(device)

    start_time = time.time()

    with torch.no_grad():
        outputs = net(selected_images)
        _, predicted = torch.max(outputs, 1)

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_samples

    print(f"CNN classified {num_samples} images in {total_time:.4f} seconds.")
    print(f"Average time per image: {avg_time:.6f} seconds.")

    return total_time, avg_time


def run_torch_tutorial(train_new):
    """ Generates the CNN model from the pytorch tutorial and returns the data, time taken and accuracy of the model.
    Saves to file for future use."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Net()
    net.to(device)
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    model_path = 'cifar_CNN_net.pth'
    start_time = time.time()

    # Check if saved model exists
    if not train_new and os.path.exists('cifar_CNN_net.pth'):
        net.load_state_dict(torch.load('cifar_CNN_net.pth'))
        net.eval()  # set to evaluation mode
        print("Loaded saved model.")
    else:
        print("Training from scratch.")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        """running model twice"""
        for epoch in range(2):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print(f"Saved trained model to {model_path}.")
        torch.save(net.state_dict(), 'cifar_CNN_net.pth')
        print('Finished Training')

    end_time = time.time()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    time_taken = end_time - start_time
    print(f"\nCNN Accuracy: {accuracy:.2f}%")
    return time_taken, accuracy, net, device
