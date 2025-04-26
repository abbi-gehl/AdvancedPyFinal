import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Need to flatten data for kNN
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

def knn_predictor(train_data, train_labels, test_data, k=3):
    preds = []
    for i in tqdm(range(test_data.size(0))):
        # euclidean distances
        distances = torch.norm(train_data - test_data[i], dim=1)
        knn_indices = distances.topk(k, largest=False).indices
        knn_labels = train_labels[knn_indices]
        pred = torch.mode(knn_labels).values.item()
        preds.append(pred)
    return preds


def imshow(img):
    img = img / 2 + 0.5     # de-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    k = 3
    print("Running KNN...")
    predictions = knn_predictor(train_data, train_labels, test_data, k=k)
    accuracy = np.mean(np.array(predictions) == test_labels.numpy())
    print(f"\nKNN Accuracy with k={k}: {accuracy * 100:.2f}%")

    # Visualize a batch of test images with predicted labels
    sample_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)
    data_iter = iter(sample_loader)
    images, labels = next(data_iter)

    imshow(torchvision.utils.make_grid(images))

    print("GroundTruth: ", ' '.join(f'{classes[labels[j]]}' for j in range(4)))
    print("Predicted:   ", ' '.join(f'{classes[predictions[j]]}' for j in range(4)))


if __name__ == '__main__':
    main()