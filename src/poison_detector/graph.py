import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.svm import OneClassSVM
from sklearn.dummy import DummyClassifier


DEBUG = os.environ.get('DEBUG', False) in ["True", "true", "1"]
device = torch.device("cuda" if torch.cuda.is_available() and not DEBUG else "cpu")

def evaluate_model(model, dataloader, labels):
    """Evaluate the model's performance.

    Args:
        model (nn.Module): The model to evaluate (discriminator in this case).
        dataloader (DataLoader): DataLoader for the dataset to evaluate on.
        labels (list): True labels for the dataset.

    Returns:
        dict: Dictionary with various metrics like accuracy, confusion matrix, etc.
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for efficiency
        all_preds = []
        # Ignore labels
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            # Thresholding: if probability > 0.5, classify as 'poisoned' (1), else 'clean' (0)
            predicted = (outputs > 0.5).float().squeeze().cpu().numpy()
            all_preds.extend(predicted)

    accuracy = accuracy_score(labels, all_preds)
    cm = confusion_matrix(labels, all_preds)
    fpr, tpr, _ = roc_curve(labels, all_preds)
    roc_auc = auc(fpr, tpr)

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc
    }

def plot_roc_curve(fpr, tpr, auc, title="ROC Curve"):
    """Plot ROC curve.

    Args:
        fpr (array): False Positive Rates
        tpr (array): True Positive Rates
        auc (float): Area Under Curve
        title (str): Title of the plot
    """
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """Plot confusion matrix.

    Args:
        cm (array): Confusion matrix
        classes (list): List of class names
        title (str): Title of the plot
        cmap (matplotlib colormap): Color map for the plot
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if cm.dtype == float else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def baseline_one_class_svm(X_train, y_train, X_test, y_test):
    """Train and evaluate One-Class SVM as a baseline.

    Args:
        X_train, X_test (array): Feature vectors for training and testing
        y_train, y_test (array): Corresponding labels

    Returns:
        dict: Similar to evaluate_model but for SVM baseline
    """
    svm_model = OneClassSVM(gamma='auto', nu=0.1).fit(X_train[y_train == 0])  # Fit only on clean data
    predictions = svm_model.predict(X_test)
    # Convert SVM's output format to match our labels
    predictions[predictions == 1] = 0  # -1 becomes 0 for normal, 1 for anomaly
    predictions[predictions == -1] = 1

    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc
    }

def baseline_most_common_class(y_test):
    """Baseline where all predictions are the most common class in the dataset.

    Args:
        y_test (array): True labels for test set.

    Returns:
        dict: Evaluation metrics for this baseline.
    """
    # Predict all as the most common class in y_test
    most_common = np.bincount(y_test).argmax()
    predictions = np.full_like(y_test, most_common)

    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    # For ROC, we'll use all predictions as the score for the positive class
    fpr, tpr, _ = roc_curve(y_test, predictions == 1)
    roc_auc = auc(fpr, tpr)

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc
    }

def baseline_random(y_test):
    """Random baseline where predictions are made randomly.

    Args:
        y_test (array): True labels for test set.

    Returns:
        dict: Evaluation metrics for this baseline.
    """
    # Random predictions
    predictions = np.random.randint(0, 2, size=len(y_test))

    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc
    }

def baseline_dummy_classifier(y_test):
    """Use scikit-learn's DummyClassifier for stratified baseline.

    Args:
        y_test (array): True labels for test set.

    Returns:
        dict: Evaluation metrics for this baseline.
    """
    dummy = DummyClassifier(strategy="stratified")
    dummy.fit(np.zeros_like(y_test), y_test)
    predictions = dummy.predict(np.zeros_like(y_test))

    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc
    }
