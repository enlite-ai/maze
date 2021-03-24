from typing import List

import numpy as np
import matplotlib.pyplot as plt


def plot_1c_image_stack(value: List[np.ndarray], **kwargs) -> None:
    """Plots a stack of single channel images with shape [N_STACK x H x W] using imshow.

    :param value: A list of image stacks.
    :param kwargs: Additional plotting relevant arguments.
    """

    # extract step key and observation name to enter appropriate plotting branch
    step_key, obs_name = kwargs["groups"]

    fig = None
    if step_key == 'step_key_0' and obs_name == 'observation-rgb2gray-resize_img':

        # randomly select one observation
        idx = np.random.random_integers(0, len(value), size=1)[0]
        obs = value[idx]
        assert obs.ndim == 3
        n_channels = obs.shape[0]
        min_val, max_val = np.min(obs), np.max(obs)

        # plot the observation
        fig = plt.figure(figsize=(max(5, 5 * n_channels), 5))
        for i, img in enumerate(obs):
            plt.subplot(1, n_channels, i+1)
            plt.imshow(img, interpolation="nearest", vmin=min_val, vmax=max_val, cmap="magma")
            plt.colorbar()

    return fig
