import numpy as np
import matplotlib.pyplot as plt


def plot_loss(training_loss_,
              val_loss_,
              num_epochs,
              path_dice_plot
              ) -> None:
    x_ticks = np.arange(1, num_epochs + 1, 1)
    plt.figure(dpi=800)
    plt.title("Loss Values")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(x_ticks, training_loss_, label="Training loss")
    plt.plot(x_ticks, val_loss_, label="Validation loss")
    plt.legend()
    plt.savefig(path_dice_plot)
    plt.close()
