import numpy as np
import matplotlib.pyplot as plt

#default plt parameters
matplotlib_params = {}

def image_grid(array, columns):
    nr, height, width, channels = array.shape
    rows = nr // columns
    assert nr == rows * columns  # otherwise not a rectangle
    result = array.reshape(rows, columns, height, width, channels) \
        .swapaxes(1, 2) \
        .reshape(height * rows, width * columns, channels)
    return result


def show_gan_image_predictions(gan, nr, columns=8, plt_params=None):
    if plt_params is None:
        plt_params = matplotlib_params
    images = gan.generate(nr)
    grid = image_grid(images, columns)
    grid = 0.5 * grid + 0.5
    plt.imshow(np.squeeze(grid), **plt_params)
    plt.show()
