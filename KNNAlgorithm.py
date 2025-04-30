import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import random
import time


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


# Need to flatten data for KNN
def flatten_loader(loader):
    data = []
    labels = []
    for images, lbls in loader:
        data.append(images.view(images.size(0), -1))  # flatten
        labels.append(lbls)
    return torch.cat(data), torch.cat(labels)


train_loader = torch.utils.data.DataLoader(train_set, batch_size=1000, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

train_data, train_labels = flatten_loader(train_loader)
test_data, test_labels = flatten_loader(test_loader)

def knn_optimization(train_data, train_labels, test_data, k):
    # vectorized algorithm
    # calculates euclidean distances of data
    dists = torch.cdist(test_data, train_data, p=2)
    # finds indices of the k closest data points for each test point
    knn_indices = dists.topk(k, largest=False).indices
    # retrieves data of those neighbors
    knn_labels = train_labels[knn_indices]
    # generates predictions
    preds = torch.mode(knn_labels, dim=1).values
    return preds.tolist()

def imshow(img):
    img = img / 2 + 0.5  # de-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def time_inference_knn(train_data, train_labels, test_data, k=3, num_samples=100):
    indices = random.sample(range(len(test_data)), num_samples)
    selected_test = test_data[indices]

    start_time = time.time()

    preds = knn_optimization(train_data, train_labels, selected_test, k=k)

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_samples

    print(f"KNN classified {num_samples} images in {total_time:.4f} seconds.")
    print(f"Average time per image: {avg_time:.6f} seconds.")

    return total_time, avg_time

def run_knn(train_new, k):
    print("Running KNN...")
    model_path = 'cifar_KNN_model.npy'

    start_time = time.time()

    if not train_new and os.path.exists(model_path):
        print("Loading saved KNN model...")
        predictions = np.load(model_path)
    else:
        print("Training KNN from scratch...")
        predictions = knn_optimization(train_data, train_labels, test_data, k)
        np.save(model_path, np.array(predictions))

    end_time = time.time()
    time_taken = end_time - start_time


    accuracy = np.mean(np.array(predictions) == test_labels.numpy())
    print(f"\nKNN Accuracy with k={k}: {accuracy * 100:.2f}%")

    return time_taken, accuracy, train_data, train_labels, test_data, test_labels
