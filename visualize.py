import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
import os

def ensure_plot_dir():
    if not os.path.exists("plot"):
        os.makedirs("plot")

def plot_confusion_matrix(y_true, y_pred, save_path=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_probs, save_path=None, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_pr_curve(y_true, y_probs, save_path=None, title="Precision-Recall Curve"):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.plot(recall, precision, color="green")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_loss(train_losses, val_losses, save_path=None, title="Train & Validation Loss"):
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()
