"""This file holds the maze random number generator as well as a Method for setting random seeds globally."""
import random

import numpy as np
import torch
import torch.backends.cudnn

from maze.utils.bcolors import BColors


def set_seeds_globally(seed: int, set_cudnn_determinism: bool, info_txt: str) -> None:
    """Set random seeds for numpy, torch and python random number generators.

    :param seed: The seed to be used.
    :param set_cudnn_determinism: Specify whether to set the cudnn backend to deterministic.
    :param info_txt: Optional info text to print with the seed info.
    """
    BColors.print_colored(f'Setting random seed globally to: {seed} - {info_txt}', BColors.OKGREEN)
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    if set_cudnn_determinism:
        if hasattr(torch, 'use_deterministic_algorithms'):
            # Pytorch version >= 1.8
            torch.use_deterministic_algorithms(True)
        elif hasattr(torch, 'set_deterministic'):
            # Pytorch version < 1.8
            torch.set_deterministic(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False


class MazeSeeding:
    """Manages the random seeding for maze.
    This class holds random state used for sampling random seed for the envs and agents.

    :param env_seed: The base_seed to be used for the envs random number generator.
    :param agent_seed: The base_seed to be used for the agents random number generator.
    :param cudnn_determinism_flag: Specify whether to set the cudnn determinism flag, this will ensure guaranty when
        working on the gpu, however some torch modules will raise runtime errors, and the processing speed will be
        decreased. For more information on this topic please refer to:
        https://pytorch.org/docs/1.7.1/notes/randomness.html?highlight=reproducability

    """
    def __init__(self, env_seed: int, agent_seed: int, cudnn_determinism_flag: bool):
        self.env_rng = np.random.RandomState(env_seed)
        self.agent_rng = np.random.RandomState(agent_seed)
        self.agent_global_seed = self.generate_agent_instance_seed()
        self.cudnn_determinism_flag = cudnn_determinism_flag

    @staticmethod
    def generate_seed_from_random_state(rng: np.random.RandomState) -> int:
        """Method for generating a random seed from the given random number generator.

        :return: Random seed.
        """
        return rng.randint(np.iinfo(np.int32).max)

    def generate_env_instance_seed(self) -> int:
        """Generate env instance seed for seeding a particular instance of the env.

        :return: A random seed for creating the env.
        """
        return self.generate_seed_from_random_state(self.env_rng)

    def generate_agent_instance_seed(self) -> int:
        """Generate agent instance seed for seeding a particular instance of the agent.

        :return: A random seed for creating the agent.
        """
        return self.generate_seed_from_random_state(self.agent_rng)

