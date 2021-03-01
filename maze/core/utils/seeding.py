import random

import numpy as np
import torch


def set_random_states(seed: int):
    """Set random states of all random generators used in the framework.

    :param: seed: the seed integer initializing the random number generators.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

