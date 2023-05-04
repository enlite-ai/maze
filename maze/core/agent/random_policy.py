"""Implements the default random policy for structured envs by sampling from the action_space."""
import itertools
from typing import Union, Dict, Tuple, Sequence, Optional

import numpy as np
from gym import spaces
from omegaconf import DictConfig

from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID
from maze.core.utils.config_utils import make_env
from maze.core.utils.seeding import MazeSeeding
from maze.core.wrappers.observation_normalization.observation_normalization_wrapper import \
    ObservationNormalizationWrapper
from maze.train.utils.train_utils import stack_numpy_dict_list
from maze.utils.process import query_cpu


class BaseRandomPolicy(Policy):
    """Implements a random structured policy that respects masking if specified.
     The policy looks for action masks in the observation with the name '<name_of_action>_mask' and samples only allowed
     actions if present.

    :param action_spaces_dict: The action_spaces dict of the env (will sample from it).
    :param do_masking: Specify whether to do masking.
    """

    def __init__(self, action_spaces_dict: Dict[Union[str, int], spaces.Space], do_masking: bool):
        self.action_spaces_dict = dict(action_spaces_dict)
        self.rng = np.random.RandomState()
        self._do_masking = do_masking

    @override(Policy)
    def seed(self, seed: int) -> None:
        """Seed the policy by setting the action space seeds."""
        self.rng = np.random.RandomState(seed)
        for key, action_space in self.action_spaces_dict.items():
            action_space.seed(MazeSeeding.generate_seed_from_random_state(self.rng))
        pass

    @override(Policy)
    def needs_state(self) -> bool:
        """This policy does not require the state() object to compute the action."""
        return False

    def _sample_random_action_for_space(self, action_space: spaces.Space, observation: ObservationType,
                                        action_key: str) -> ActionType:
        """Sample a random action from the given space.

        :param action_space: The action space to sample from.
        :param observation: The current observation.
        :param action_key: The corresponding action key.
        :return: A sampled action.
        """
        assert not isinstance(action_space, spaces.Dict)

        action_mask_name = f'{action_key}_mask'
        if (not self._do_masking or action_mask_name not in observation or np.all(observation[action_mask_name] == 0) or
                np.all(observation[action_mask_name] == 1)):
            return action_space.sample()

        assert action_mask_name in observation
        assert np.all((observation[action_mask_name] == 0) | (observation[action_mask_name] == 1))
        assert np.any(observation[action_mask_name] == 1)

        if isinstance(action_space, spaces.Discrete):
            possible_actions = np.where(observation[action_mask_name] == 1)[0]
            return self.rng.choice(possible_actions)
        elif isinstance(action_space, spaces.MultiBinary):
            action = action_space.sample()
            action[~observation[action_mask_name].astype(bool)] = 0
            return action
        else:
            raise NotImplementedError(f'Masked random policy is not yet implemented for {type(action_space)} '
                                      f'spaces.')

    @override(Policy)
    def compute_action(self,
                       observation: ObservationType,
                       maze_state: Optional[MazeStateType],
                       env: Optional[BaseEnv] = None,
                       actor_id: Optional[ActorID] = None,
                       deterministic: bool = False) -> ActionType:
        """Sample random action from the given action space."""
        if actor_id:
            action_space = self.action_spaces_dict[actor_id.step_key]
        else:
            assert len(self.action_spaces_dict) == 1, "action spaces for multiple sub-steps are available, please " \
                                                      "specify actor ID explicitly"
            action_space = list(self.action_spaces_dict.values())[0]

        assert isinstance(action_space, spaces.Dict)
        action = {kk: self._sample_random_action_for_space(space, observation, kk) for kk, space in
                  action_space.spaces.items()}

        return action

    def _try_n_samples_without_replace(self, num_candidates: Optional[int], actor_id: Optional[ActorID],
                                       observation: ObservationType) -> Tuple[
        bool, Optional[Sequence[ActionType]]]:
        """Try to compute n samples without replacement from the action spaces. This is only works if there is only
        one discrete action space to sample from.

        :param num_candidates: The number of candidates ot retrieve.
        :param actor_id: The current actor id.
        :param observation: The current observation.
        :return: True iff it was possible to sample the actions and the action candidates.

        """
        if actor_id:
            step_key = actor_id.step_key
        else:
            assert len(self.action_spaces_dict) == 1, "action spaces for multiple sub-steps are available, please " \
                                                      "specify actor ID explicitly"
            step_key = list(self.action_spaces_dict.keys())[0]
        action_space = self.action_spaces_dict[step_key]

        assert isinstance(action_space, spaces.Dict)

        if not all([isinstance(space, spaces.Discrete) for space in action_space.spaces.values()]):
            return False, None

        options = list()
        for action_key, action_sub_space in action_space.spaces.items():
            options_ss = np.arange(action_sub_space.n)
            action_mask_name = f'{action_key}_mask'
            if self._do_masking and action_mask_name in observation:
                assert np.all((observation[action_mask_name] == 0) | (observation[action_mask_name] == 1))
                options_ss = np.where(observation[action_mask_name] == 1)[0]
            options.append(list(options_ss))

        num_options = np.product(list(map(len, options)))
        if num_options > 10000:
            # Too many options for complete enumerations
            return False, None

        num_candidates = num_options if num_candidates is None else num_candidates
        complete_options = list(itertools.product(*options))
        candidates_idx = self.rng.permutation(self.rng.choice(num_options, size=num_candidates, replace=False))
        candidates = [complete_options[idx] for idx in candidates_idx][:num_candidates]

        return True, [{action_key: vv for action_key, vv in zip(action_space, cc)} for cc in candidates]

    @override(Policy)
    def compute_top_action_candidates(self, observation: ObservationType, num_candidates: Optional[int],
                                      maze_state: Optional[MazeStateType], env: Optional[BaseEnv],
                                      actor_id: ActorID = None) \
            -> Tuple[Sequence[ActionType], Sequence[float]]:
        """Sample multiple random actions from the provided action space (and assign uniform probabilities
        to the sampled actions)."""

        # Try to sample the candidates without replacement (only works if there is only one discrete action space).
        success, candidates = self._try_n_samples_without_replace(num_candidates, actor_id, observation)

        if not success:
            candidates = []
            assert num_candidates is not None
            for _ in range(num_candidates):
                candidates.append(
                    self.compute_action(observation=observation, maze_state=maze_state, env=env, actor_id=actor_id)
                )

        return candidates, [1.0 / len(candidates)] * len(candidates)


class RandomPolicy(BaseRandomPolicy):
    """Implements a random structured policy.

    :param action_spaces_dict: The action_spaces dict of the env (will sample from it).
    """

    def __init__(self, action_spaces_dict: Dict[Union[str, int], spaces.Space]):
        super().__init__(action_spaces_dict, do_masking=False)


class MaskedRandomPolicy(BaseRandomPolicy):
    """Implements a masked random structured policy.
     The policy looks for action masks in the observation with the name '<name_of_action>_mask' and samples only allowed
     actions if present.

    :param action_spaces_dict: The action_spaces dict of the env (will sample from it).
    """

    def __init__(self, action_spaces_dict: Dict[Union[str, int], spaces.Space]):
        super().__init__(action_spaces_dict, do_masking=True)


class DistributedRandomPolicy(RandomPolicy):
    """Implements random structured policy with the ability to sample multiple actions simultaneously.

    Mimics behavior of a Torch policy in batched distributed environment scenarios,
    when actions are sampled for all environments at once. (Useful mostly for testing purposes.)

    :param action_spaces_dict: The action_spaces dict of the env (will sample from it)
    :param concurrency: How many actions to sample at once. Should correspond to concurrency of the distributed env.
    """

    def __init__(self, action_spaces_dict: Dict[Union[str, int], spaces.Space], concurrency: int):
        super().__init__(action_spaces_dict)
        self.concurrency = concurrency if concurrency > 0 else query_cpu()

    def compute_action(self,
                       observation: ObservationType,
                       maze_state: Optional[MazeStateType],
                       env: Optional[BaseEnv] = None,
                       actor_id: Optional[ActorID] = None,
                       deterministic: bool = False) -> ActionType:
        """Sample multiple actions together."""
        if actor_id:
            action_space = self.action_spaces_dict[actor_id.step_key]
        else:
            assert len(self.action_spaces_dict) == 1, "action spaces for multiple sub-steps are available, please " \
                                                      "specify actor ID explicitly"
            action_space = list(self.action_spaces_dict.values())[0]

        return stack_numpy_dict_list([action_space.sample() for _ in range(self.concurrency)])


def random_policy_from_config(env_config: DictConfig, wrappers_config: DictConfig) -> RandomPolicy:
    """Factory function for instantiating a :class:`~maze.core.agent.structured_policy.RandomPolicy`
    from an env configuration.

    :param env_config: Environment config
    :param wrappers_config: Wrapper config
    :return: A random structured policy.
    """
    env = make_env(env_config, wrappers_config)
    # observation normalization needs to be deactivated when no stats are provided
    if isinstance(env, ObservationNormalizationWrapper):
        env.set_observation_collection(True)
    action_spaces_dict = env.action_spaces_dict
    env.close()
    return RandomPolicy(action_spaces_dict)


def masked_random_policy_from_config(env_config: DictConfig, wrappers_config: DictConfig) -> MaskedRandomPolicy:
    """Factory function for instantiating a :class:`~maze.core.agent.structured_policy.RandomPolicy`
    from an env configuration.

    :param env_config: Environment config
    :param wrappers_config: Wrapper config
    :return: A random structured policy.
    """
    env = make_env(env_config, wrappers_config)
    # observation normalization needs to be deactivated when no stats are provided
    if isinstance(env, ObservationNormalizationWrapper):
        env.set_observation_collection(True)
    action_spaces_dict = env.action_spaces_dict
    env.close()
    return MaskedRandomPolicy(action_spaces_dict)
