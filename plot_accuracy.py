"""
plot_accuracy.py — Training history visualisation utilities.
"""

import matplotlib.pyplot as plt


def plot_accuracy(history):
    """Plot training and validation accuracy curves."""
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'],     label='Train accuracy')
    plt.plot(history.history['val_accuracy'], label='Val accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.title('Model Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def plot_loss(history):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'],     label='Train loss')
    plt.plot(history.history['val_loss'], label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_history(history):
    """Plot both accuracy and loss side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(history.history['accuracy'],     label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim([0, 1])
    axes[0].legend()

    axes[1].plot(history.history['loss'],     label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
