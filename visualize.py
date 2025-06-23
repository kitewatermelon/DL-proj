import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
import os

def ensure_plot_dir():
    if not os.path.exists("plot"):
        os.makedirs("plot")

def plot_confusion_matrix(y_true, y_pred, save_path=None, title="Confusion Matrix"):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_probs, save_path=None, title="ROC Curve"):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color="darkorange", label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_pr_curve(y_true, y_probs, save_path=None, title="Precision-Recall Curve"):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.plot(recall, precision, color="green")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_loss_curve(train_losses, val_losses, save_path=None, title="Loss Curve"):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()