"""This file holds the maze random number generator as well as a Method for setting random seeds globally."""
import random
from typing import Optional, Sequence, Any, Union, List

import numpy as np
import torch
import torch.backends.cudnn

from maze.core.utils.factory import ConfigType, Factory
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

    def __init__(self, env_seed: int, agent_seed: int, cudnn_determinism_flag: bool,
                 explicit_env_seeds: Optional[Union[Sequence[Any], ConfigType]],
                 explicit_agent_seeds: Optional[Union[Sequence[Any], ConfigType]],
                 shuffle_seeds: bool):
        self._env_base_seed = env_seed
        self._agent_base_seed = agent_seed
        self.env_rng = np.random.RandomState(env_seed)
        self.agent_rng = np.random.RandomState(agent_seed)
        self.cudnn_determinism_flag = cudnn_determinism_flag

        self._shuffle_seeds = shuffle_seeds

        self._explicit_env_seeds = explicit_env_seeds
        if self._explicit_env_seeds is not None:
            self._explicit_env_seeds = list(Factory(Sequence).instantiate(explicit_env_seeds))
            if self._shuffle_seeds:
                seed_is_int = isinstance(self._explicit_env_seeds[0], int)
                self._explicit_env_seeds = list(self.env_rng.permutation(self._explicit_env_seeds))
                if seed_is_int:
                    self._explicit_env_seeds = list(map(int, self._explicit_env_seeds))

        self._explicit_agent_seeds = explicit_agent_seeds
        if self._explicit_agent_seeds is not None:
            self._explicit_agent_seeds = list(Factory(Sequence).instantiate(explicit_agent_seeds))
            if self._shuffle_seeds:
                seed_is_int = isinstance(self._explicit_agent_seeds[0], int)
                self._explicit_agent_seeds = list(self.agent_rng.permutation(self._explicit_agent_seeds))
                if seed_is_int:
                    self._explicit_agent_seeds = list(map(int, self._explicit_agent_seeds))

        self.global_seed = self.generate_agent_instance_seed()

    def get_explicit_env_seeds(self, n_seeds: int) -> List[Any]:
        """Return a list of explicit env seeds to be used for each episode.

        :param n_seeds: The number of seeds to be returned.
        :return: A list of seeds.
        """
        if self._explicit_env_seeds is not None:
            return self._explicit_env_seeds[:n_seeds]
        else:
            seeds = [self.generate_env_instance_seed() for _ in range(n_seeds)]
            if self._shuffle_seeds:
                seeds = list(map(int, self.env_rng.permutation(seeds)))
            return seeds

    def get_explicit_agent_seeds(self, n_seeds: int) -> List[Any]:
        """Return a list of explicit agent seeds to be used for each episode.

        :param n_seeds: The number of seeds to be returned.
        :return: A list of seeds.
        """
        if self._explicit_agent_seeds is not None:
            return self._explicit_agent_seeds[:n_seeds]
        else:
            seeds = [self.generate_agent_instance_seed() for _ in range(n_seeds)]
            if self._shuffle_seeds:
                seeds = list(map(int, self.agent_rng.permutation(seeds)))
            return seeds

    def get_env_base_seed(self) -> int:
        """Return the env base seed.

        :return: The env base seed.
        """
        return self._env_base_seed

    def get_agent_base_seed(self) -> int:
        """Return the agent base seed.

        :return: The agent base seed.
        """
        return self._agent_base_seed

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
