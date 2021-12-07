from typing import Tuple, Optional, Sequence, Dict, List

import torch
import torch.nn as nn
from maze.core.agent.policy import Policy
from maze.core.agent.torch_model import TorchModel
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import StructuredEnv, ActorID
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.models.built_in.flatten_concat import FlattenConcatPolicyNet
from maze.train.trainers.es.distributed.es_dummy_distributed_rollouts import ESDummyDistributedRollouts
from maze.train.trainers.es.distributed.es_subproc_distributed_rollouts import ESSubprocDistributedRollouts
from maze.train.trainers.es.es_algorithm_config import ESAlgorithmConfig
from maze.train.trainers.es.es_shared_noise_table import SharedNoiseTable
from maze.train.trainers.es.es_trainer import ESTrainer
from maze.train.trainers.es.optimizers.adam import Adam


class DummyPolicyWrapper(Policy, TorchModel):
    """A dummy implementation of a policy that wraps a TorchPolicy, as supported by ES."""

    def __init__(self, torch_policy: TorchPolicy):
        self.torch_policy = torch_policy
        super().__init__(device=torch_policy.device)

    @override(Policy)
    def seed(self, seed: int) -> None:
        """Seed the policy to be used."""

    @override(Policy)
    def needs_state(self) -> bool:
        """Env state not required by `compute_action`"""
        return False

    @override(Policy)
    def needs_env(self) -> bool:
        """We want to access the env in `compute_action`"""
        return True

    @override(Policy)
    def compute_action(self, observation: ObservationType, maze_state: Optional[MazeStateType], env: MazeEnv,
                       actor_id: Optional[ActorID] = None, deterministic: bool = False) -> ActionType:
        """Here we could do arbitrarily complex, non-differentiable processing on top of the policy."""
        actions, probs = self.torch_policy.compute_top_action_candidates(
            observation=observation, maze_state=maze_state, env=env,
            actor_id=actor_id,
            num_candidates=2)

        # let's pick the action dependent on the env time
        # (for demonstration only, not a sensible logic, especially if there are only 2 actions as in cartpole)
        return actions[env.get_env_time() % 2]

    @override(Policy)
    def compute_top_action_candidates(self, observation: ObservationType, num_candidates: Optional[int],
                                      maze_state: Optional[MazeStateType], env: Optional[BaseEnv],
                                      actor_id: Optional[ActorID] = None) \
            -> Tuple[Sequence[ActionType], Sequence[float]]:
        """Not supported"""
        raise NotImplementedError

    @override(TorchModel)
    def parameters(self) -> List[torch.Tensor]:
        """Forward the method call to the wrapped TorchPolicy"""
        return self.torch_policy.parameters()

    @override(TorchModel)
    def eval(self) -> None:
        """Forward the method call to the wrapped TorchPolicy"""
        self.torch_policy.eval()

    @override(TorchModel)
    def train(self) -> None:
        """Forward the method call to the wrapped TorchPolicy"""
        self.torch_policy.train()

    @override(TorchModel)
    def to(self, device: str) -> None:
        """Forward the method call to the wrapped TorchPolicy"""
        self.torch_policy.to(device)

    @override(TorchModel)
    def state_dict(self) -> Dict:
        """Forward the method call to the wrapped TorchPolicy"""
        return self.torch_policy.state_dict()

    @override(TorchModel)
    def load_state_dict(self, state_dict: Dict) -> None:
        """Forward the method call to the wrapped TorchPolicy"""
        self.torch_policy.load_state_dict(state_dict)


def train_setup(n_epochs: int, policy_wrapper=None) -> Tuple[TorchPolicy, StructuredEnv, ESTrainer]:
    """Trains the cart pole environment with the multi-step a2c implementation.
    """

    # initialize distributed env
    env = GymMazeEnv(env="CartPole-v0")

    # initialize distribution mapper
    distribution_mapper = DistributionMapper(action_space=env.action_space, distribution_mapper_config={})

    # initialize policies
    policies = {0: FlattenConcatPolicyNet({'observation': (4,)}, {'action': (2,)}, hidden_units=[16], non_lin=nn.Tanh)}

    # initialize optimizer
    policy = TorchPolicy(networks=policies, distribution_mapper=distribution_mapper, device="cpu")

    # reduce the noise table size to speed up testing
    shared_noise = SharedNoiseTable(count=1_000_000)

    algorithm_config = ESAlgorithmConfig(
        n_rollouts_per_update=100,
        n_timesteps_per_update=0,
        max_steps=0,
        optimizer=Adam(step_size=0.01),
        l2_penalty=0.005,
        noise_stddev=0.02,
        n_epochs=n_epochs,
        policy_wrapper=policy_wrapper
    )

    # train agent
    trainer = ESTrainer(algorithm_config=algorithm_config,
                        shared_noise=shared_noise,
                        torch_policy=policy,
                        normalization_stats=None)

    return policy, env, trainer


def test_es():
    policy, env, trainer = train_setup(n_epochs=2)

    trainer.train(
        ESDummyDistributedRollouts(env=env, n_eval_rollouts=2, shared_noise=trainer.shared_noise,
                                   agent_instance_seed=1234), model_selection=None)


def test_policy_wrapper():
    policy, env, trainer = train_setup(n_epochs=2, policy_wrapper={"_target_": DummyPolicyWrapper})

    trainer.train(
        ESDummyDistributedRollouts(env=env, n_eval_rollouts=2, shared_noise=trainer.shared_noise,
                                   agent_instance_seed=1234), model_selection=None)


def test_subproc_distributed_rollouts():
    policy, env, trainer = train_setup(n_epochs=2)

    rollouts = ESSubprocDistributedRollouts(
        env_factory=lambda: GymMazeEnv(env="CartPole-v0"),
        n_training_workers=2,
        n_eval_workers=1,
        shared_noise=trainer.shared_noise,
        env_seeds=[1337] * 3,
        agent_seed=1337
    )

    trainer.train(rollouts, model_selection=None)


def test_subproc_distributed_rollouts_with_policy_wrapper():
    policy, env, trainer = train_setup(n_epochs=2, policy_wrapper={"_target_": DummyPolicyWrapper})

    rollouts = ESSubprocDistributedRollouts(
        env_factory=lambda: GymMazeEnv(env="CartPole-v0"),
        n_training_workers=2,
        n_eval_workers=1,
        shared_noise=trainer.shared_noise,
        env_seeds=[1337] * 3,
        agent_seed=1337
    )

    trainer.train(rollouts, model_selection=None)
