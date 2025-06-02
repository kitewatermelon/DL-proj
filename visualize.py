import matplotlib.pyplot as plt
import os

def plot_loss_accuracy(train_losses, valid_losses, valid_accuracies, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, valid_losses, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over epochs")

    plt.subplot(1,2,2)
    plt.plot(epochs, valid_accuracies, label="Valid Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy over epochs")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_valid_plot.png"))
    plt.close()
