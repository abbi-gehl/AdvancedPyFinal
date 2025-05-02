import KNNAlgorithm
import TorchTutorialModel
import matplotlib.pyplot as plt
import numpy as np
import torch
import random


def gen_images(net, device, train_data, train_labels, test_data, test_labels, k):
    """ image function that generates and plots 4 images and tests the classification of both models """
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # CIFAR-10 classes

    sample_size = 5  # Number of random images to classify
    indices = random.sample(range(len(test_data)), sample_size)

    images_flat = test_data[indices]
    true_labels = test_labels[indices]

    images = images_flat.reshape(-1, 3, 32, 32)
    images_device = images.to(device)

    # CNN predictions
    net.eval()
    with torch.no_grad():
        outputs = net(images_device)
        _, cnn_preds = torch.max(outputs, 1)

    # KNN predictions
    from KNNAlgorithm import knn_optimization
    knn_preds = knn_optimization(train_data, train_labels, images_flat, k)

    # Plot
    fig, axs = plt.subplots(1, sample_size, figsize=(15, 5))
    for i in range(sample_size):
        img = images[i]
        img = img / 2 + 0.5  # denormalize
        npimg = img.cpu().numpy()

        axs[i].imshow(np.transpose(npimg, (1, 2, 0)))
        axs[i].axis('off')

        cnn_correct = cnn_preds[i].item() == true_labels[i].item()
        knn_correct = knn_preds[i] == true_labels[i].item()

        cnn_color = 'green' if cnn_correct else 'red'
        knn_color = 'green' if knn_correct else 'red'

        axs[i].set_title(
            f"CNN: {classes[cnn_preds[i]]}", color=cnn_color, fontsize=9
        )
        axs[i].text(
            0.5, -0.15, f"KNN: {classes[knn_preds[i]]}",
            color=knn_color,
            fontsize=9,
            ha='center',
            va='center',
            transform=axs[i].transAxes
        )

    plt.tight_layout()
    plt.show()


def main():
    """ Main function that runs the algorithms and plots the results"""
    k=5
    train_new = False

    torch_time, torch_accuracy, net, device = TorchTutorialModel.run_torch_tutorial(train_new)
    knn_time, knn_accuracy, train_data, train_labels, test_data, test_labels = KNNAlgorithm.run_knn(train_new, k)

    cnn_inference_time, cnn_avg = TorchTutorialModel.time_inference_cnn(net, device, TorchTutorialModel.testloader)
    knn_inference_time, knn_avg = KNNAlgorithm.time_inference_knn(train_data, train_labels, test_data, k)

    models = ['CNN (Torch)', f'KNN k= {k}']
    times = [torch_time, knn_time]
    accuracies = [torch_accuracy, knn_accuracy*100]
    inference_times = [cnn_avg, knn_avg]
    if train_new:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].bar(models, times, color=['skyblue', 'lightgreen'])
        axs[0].set_title('Computing Time Comparison')
        axs[0].set_ylabel('Time (seconds)')

        axs[1].bar(models, accuracies, color=['skyblue', 'lightgreen'])
        axs[1].set_title('Accuracy Comparison')
        axs[1].set_ylabel('Accuracy (%)')
        axs[1].set_ylim(0, 100)

        axs[2].bar(models, inference_times, color=['skyblue', 'lightgreen'])
        axs[2].set_title('Avg Inference Time (100 samples)')
        axs[2].set_ylabel('Time (seconds)')

        plt.tight_layout()
        plt.show()
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].bar(models, accuracies, color=['skyblue', 'lightgreen'])
        axs[0].set_title('Accuracy Comparison')
        axs[0].set_ylabel('Accuracy (%)')
        axs[0].set_ylim(0, 100)

        print("testing algorithms")

        axs[1].bar(models, inference_times, color=['skyblue', 'lightgreen'])
        axs[1].set_title('Avg Inference Time (100 samples)')
        axs[1].set_ylabel('Time (seconds)')

        plt.tight_layout()
        plt.show()
        print("Generating images...")
        gen_images(net, device, train_data, train_labels, test_data, test_labels, k)



if __name__ == '__main__':
    main()