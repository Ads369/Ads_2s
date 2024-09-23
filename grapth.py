import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from keras.callbacks import History
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

def show_history(history: History, figsize: Tuple[float, float] = (12, 10), dpi: int = 100) -> None:
    """
    Plot training history and detect overfitting.

    Args:
    history (History): Keras History object containing training history.
    figsize (Tuple[float, float]): Figure size (width, height) in inches.
    dpi (int): Dots per inch for the figure.
    """
    history_dict = history.history

    required_keys = ['acc', 'val_acc', 'loss', 'val_loss']
    if not all(key in history_dict for key in required_keys):
        raise ValueError(f"History dictionary is missing one or more required keys: {required_keys}")

    epochs = range(1, len(history_dict['acc']) + 1)

    with plt.style.context('seaborn'):
        fig: Figure
        ax1: Axes
        ax2: Axes
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=dpi)
        fig.suptitle('Model Training History', fontsize=16, fontweight='bold')

        # Accuracy plot
        ax1.plot(epochs, history_dict['acc'], 'r-', label='Training Accuracy', linewidth=2)
        ax1.plot(epochs, history_dict['val_acc'], 'b-', label='Validation Accuracy', linewidth=2)
        ax1.set_title('Accuracy', fontsize=14)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Loss plot
        ax2.plot(epochs, history_dict['loss'], 'r-', label='Training Loss', linewidth=2)
        ax2.plot(epochs, history_dict['val_loss'], 'b-', label='Validation Loss', linewidth=2)
        ax2.set_title('Loss', fontsize=14)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)

        # Detect overfitting
        val_loss = np.array(history_dict['val_loss'])
        min_loss_epoch = np.argmin(val_loss) + 1

        if min_loss_epoch < len(epochs):
            ax2.axvline(x=min_loss_epoch, color='g', linestyle='--', label='Potential Overfitting Point')
            ax2.text(min_loss_epoch, ax2.get_ylim()[1], f'Epoch {min_loss_epoch}',
                     horizontalalignment='center', verticalalignment='bottom', color='g')
            ax2.legend(loc='upper right', fontsize=10)

            print(f"Potential overfitting detected at epoch {min_loss_epoch}")

        plt.tight_layout()
        plt.show()



def show_history_2(store):
    acc = store.history['acc']
    val_acc = store.history['val_acc']
    loss = store.history['loss']
    val_loss = store.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Detecting overfitting: where validation accuracy decreases while training accuracy increases
    overfit_epoch = None
    for i in range(1, len(val_acc)):
        if val_acc[i] < val_acc[i - 1] and acc[i] > acc[i - 1]:
            overfit_epoch = i + 1  # Epoch is 1-based

    # Set a nice style for the plots
    plt.style.use('seaborn-darkgrid')

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, acc, 'r-', marker='o', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_acc, 'b--', marker='s', label='Validation Accuracy', linewidth=2)

    if overfit_epoch:
        plt.axvline(overfit_epoch, color='purple', linestyle='--', label=f'Overfitting Detected (Epoch {overfit_epoch})')

    plt.title('Accuracy on Training and Validation Data', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True)

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'r-', marker='o', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'b--', marker='s', label='Validation Loss', linewidth=2)

    if overfit_epoch:
        plt.axvline(overfit_epoch, color='purple', linestyle='--', label=f'Overfitting Detected (Epoch {overfit_epoch})')

    plt.title('Loss on Training and Validation Data', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True)

    plt.show()
