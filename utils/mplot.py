import io

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def plotConfusionMatrix(cm, name: str = "", labels=["outdoor", "indoor"]):
    # check if tensor or not
    if isinstance(cm, torch.Tensor):
        cm = cm.cpu().numpy()
    # use blue theme
    fig = plt.figure(figsize=(10, 10), dpi=512)
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    # add the values on the grid
    for (i, j), z in np.ndenumerate(cm):
        # invert the color of the text if it is too dark
        if z < 0.5:
            ax.text(
                j, i, "{:0.1f} %".format(z * 100), ha="center", va="center", fontsize=20
            )
        else:
            ax.text(
                j,
                i,
                "{:0.1f} %".format(z * 100),
                ha="center",
                va="center",
                fontsize=20,
                color="white",
            )
    fig.colorbar(cax)
    plt.title(f"{name} Confusion Matrix ", fontsize=25, fontweight="bold")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.xaxis.tick_bottom()
    ax.set_xticklabels(
        labels,
        # rotation=45,
        fontsize=20,
    )
    ax.set_yticklabels(
        labels,
        rotation=90,
        fontsize=20,
    )
    plt.xlabel(
        "Predicted",
        fontsize=15,
    )
    plt.ylabel(
        "Ground Truth",
        fontsize=15,
    )
    plt.tight_layout()
    fig.canvas.draw()
    return fig


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    # image = tf.image.decode_png(buf.getvalue(), channels=4)
    # use torch
    image = torch.from_numpy(np.array(Image.open(buf)))
    # convert to CHW
    image = image.permute(2, 0, 1)
    return image
