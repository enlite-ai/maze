"""Test the trainer of the SAC implementation"""
from collections import OrderedDict
from typing import Dict, Union, Sequence

import torch.nn as nn
from gym import spaces

from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.agent.torch_state_action_critic import TorchStepStateActionCritic
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.general.concat import ConcatenationBlock
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.perception.perception_utils import observation_spaces_to_in_shapes
from maze.perception.weight_init import make_module_init_normc
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator
from maze.train.parallelization.distributed_actors.dummy_distributed_workers_with_buffer import \
    DummyDistributedWorkersWithBuffer
from maze.train.trainers.common.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from maze.train.trainers.sac.sac_algorithm_config import SACAlgorithmConfig
from maze.train.trainers.sac.sac_runners import SACRunner
from maze.train.trainers.sac.sac_trainer import SAC


class PolicyNet(nn.Module):
    """Simple feed forward policy network.
    """

    def __init__(self, obs_shapes: Dict[str, Sequence[int]], action_logits_shapes: Dict[str, Sequence[int]],
                 non_lin: Union[str, type(nn.Module)]):
        super().__init__()
        self.obs_shapes = obs_shapes
        action_key = list(action_logits_shapes.keys())[0]
        # build perception part
        self.perception_dict = OrderedDict()
        self.perception_dict['embedding'] = DenseBlock(in_keys="observation", out_keys="embedding",
                                                       in_shapes=obs_shapes['observation'],
                                                       hidden_units=[256, 256], non_lin=non_lin)

        # build action head
        self.perception_dict[action_key] = LinearOutputBlock(in_keys="embedding", out_keys=action_key,
                                                             in_shapes=self.perception_dict[
                                                                 'embedding'].out_shapes(),
                                                             output_units=action_logits_shapes[action_key][0])

        self.perception_net = InferenceBlock(
            in_keys='observation', out_keys=action_key,
            in_shapes=[self.obs_shapes['observation']],
            perception_blocks=self.perception_dict)

        # initialize model weights
        self.perception_net.apply(make_module_init_normc(1.0))
        self.perception_dict[action_key].apply(make_module_init_normc(0.01))

    def forward(self, x):
        """ forward pass. """
        return self.perception_net(x)


class QCriticNetContinuous(nn.Module):
    """Simple Q critic for mixed action heads (that is not all discrete). As such it computes a single q_value output
    for all observations.
    """

    def __init__(self, obs_shapes: Dict[str, Sequence[int]], action_spaces_dict: Dict[Union[str, int], spaces.Space],
                 non_lin: Union[str, type(nn.Module)]):
        super().__init__()
        self.obs_shapes = obs_shapes
        # build perception part
        self.perception_dict = OrderedDict()
        self.perception_dict['latent-obs'] = DenseBlock(in_keys="observation", out_keys="latent-obs",
                                                        in_shapes=obs_shapes['observation'],
                                                        hidden_units=[256], non_lin=non_lin)
        self.perception_dict['latent-act'] = DenseBlock(in_keys="action", out_keys="latent-act",
                                                        in_shapes=obs_shapes['action'],
                                                        hidden_units=[256], non_lin=non_lin)

        self.perception_dict['concat'] = ConcatenationBlock(
            in_keys=['latent-obs', 'latent-act'],
            in_shapes=self.perception_dict['latent-obs'].out_shapes() +
                      self.perception_dict['latent-act'].out_shapes(), concat_dim=-1, out_keys='concat'
        )

        self.perception_dict['latent'] = DenseBlock(in_keys="concat", out_keys="latent",
                                                    in_shapes=self.perception_dict['concat'].out_shapes(),
                                                    hidden_units=[256], non_lin=non_lin)

        # build action head
        self.perception_dict['q_value'] = LinearOutputBlock(in_keys="latent", out_keys="q_value",
                                                            in_shapes=self.perception_dict[
                                                                'latent'].out_shapes(),
                                                            output_units=1)

        self.perception_net = InferenceBlock(
            in_keys=['observation', 'action'], out_keys='q_value',
            in_shapes=[self.obs_shapes['observation'], self.obs_shapes['action']],
            perception_blocks=self.perception_dict)

        # initialize model weights
        self.perception_net.apply(make_module_init_normc(1.0))
        self.perception_dict['q_value'].apply(make_module_init_normc(0.01))

    def forward(self, x):
        """ forward pass. """
        return self.perception_net(x)


def train_function(n_epochs: int, epoch_length: int, deterministic_eval: bool,
                   eval_repeats: int, distributed_env_cls, split_rollouts_into_transitions: bool) -> SAC:
    """Implements the lunar lander continuous env and performs tests on it w.r.t. the sac trainer.
    """

    # initialize distributed env
    env_factory = lambda: GymMazeEnv(env="LunarLanderContinuous-v2")

    # initialize the env and enable statistics collection
    eval_env = distributed_env_cls([env_factory for _ in range(2)],
                                   logging_prefix='eval')

    env = env_factory()
    # init distribution mapper
    distribution_mapper = DistributionMapper(
        action_space=env.action_space,
        distribution_mapper_config=[{'action_space': 'gym.spaces.Box',
                                     'distribution': 'maze.distributions.squashed_gaussian.SquashedGaussianProbabilityDistribution'}])

    action_shapes = {step_key: {action_head: tuple(distribution_mapper.required_logits_shape(action_head))
                                for action_head in env.action_spaces_dict[step_key].spaces.keys()}
                     for step_key in env.action_spaces_dict.keys()}

    obs_shapes = observation_spaces_to_in_shapes(env.observation_spaces_dict)
    # initialize policies
    policies = {ii: PolicyNet(obs_shapes=obs_shapes[ii],
                              action_logits_shapes=action_shapes[ii], non_lin=nn.Tanh) for ii in obs_shapes.keys()}

    for key, value in env.action_spaces_dict.items():
        for act_key, act_space in value.spaces.items():
            obs_shapes[key][act_key] = act_space.sample().shape
    # initialize critic
    critics = {ii: QCriticNetContinuous(obs_shapes[ii], non_lin=nn.Tanh,
                                        action_spaces_dict=env.action_spaces_dict) for ii in obs_shapes.keys()}

    # initialize optimizer
    algorithm_config = SACAlgorithmConfig(
        n_rollout_steps=5, lr=0.001, entropy_coef=0.2,
        gamma=0.99, max_grad_norm=0.5, batch_size=100,
        num_actors=2, tau=0.005,
        target_update_interval=1,
        entropy_tuning=False, device='cpu',
        replay_buffer_size=10000, initial_buffer_size=100,
        initial_sampling_policy={'_target_': 'maze.core.agent.random_policy.RandomPolicy'},
        rollouts_per_iteration=1, split_rollouts_into_transitions=split_rollouts_into_transitions,
        entropy_coef_lr=0.0007, num_batches_per_iter=1, n_epochs=n_epochs, epoch_length=epoch_length,
        rollout_evaluator=RolloutEvaluator(
            eval_env=eval_env,
            n_episodes=eval_repeats,
            model_selection=None,
            deterministic=deterministic_eval
        ),
        patience=50,
        target_entropy_multiplier=1.0
    )

    actor_policy = TorchPolicy(networks=policies, distribution_mapper=distribution_mapper, device='cpu')

    replay_buffer = UniformReplayBuffer(buffer_size=algorithm_config.replay_buffer_size, seed=1234)
    SACRunner.init_replay_buffer(replay_buffer=replay_buffer, initial_sampling_policy=algorithm_config.initial_sampling_policy,
                                 initial_buffer_size=algorithm_config.initial_buffer_size, replay_buffer_seed=1234,
                                 split_rollouts_into_transitions=split_rollouts_into_transitions,
                                 n_rollout_steps=algorithm_config.n_rollout_steps, env_factory=env_factory)
    distributed_actors = DummyDistributedWorkersWithBuffer(
        env_factory=env_factory, worker_policy=actor_policy, n_rollout_steps=algorithm_config.n_rollout_steps,
        n_workers=algorithm_config.num_actors, batch_size=algorithm_config.batch_size,
        rollouts_per_iteration=algorithm_config.rollouts_per_iteration,
        split_rollouts_into_transitions=split_rollouts_into_transitions,
        env_instance_seeds=list(range(algorithm_config.num_actors)), replay_buffer=replay_buffer)

    critics_policy = TorchStepStateActionCritic(networks=critics, num_policies=1, device='cpu',
                                                only_discrete_spaces={0: False},
                                                action_spaces_dict=env.action_spaces_dict)

    learner_model = TorchActorCritic(policy=actor_policy,
                                     critic=critics_policy,
                                     device='cpu')

    # initialize trainer
    sac = SAC(
        learner_model=learner_model,
        distributed_actors=distributed_actors,
        algorithm_config=algorithm_config,
        evaluator=algorithm_config.rollout_evaluator,
        model_selection=None
    )

    # train agent
    sac.train(n_epochs=algorithm_config.n_epochs)

    return sac


def test_sac_trainer_keeping_rollouts():
    """ sac unit tests """
    sac = train_function(n_epochs=2, epoch_length=2, deterministic_eval=False,
                         eval_repeats=1, distributed_env_cls=SequentialVectorEnv,
                         split_rollouts_into_transitions=False)
    assert isinstance(sac, SAC)

    sac = train_function(n_epochs=2, epoch_length=2, deterministic_eval=False,
                         eval_repeats=0, distributed_env_cls=SequentialVectorEnv,
                         split_rollouts_into_transitions=False)
    assert isinstance(sac, SAC)


def test_sac_trainer_splitting_rollouts():
    """ sac unit tests """
    sac = train_function(n_epochs=2, epoch_length=2, deterministic_eval=False,
                         eval_repeats=1, distributed_env_cls=SequentialVectorEnv,
                         split_rollouts_into_transitions=True)
    assert isinstance(sac, SAC)

    sac = train_function(n_epochs=2, epoch_length=2, deterministic_eval=False,
                         eval_repeats=0, distributed_env_cls=SequentialVectorEnv,
                         split_rollouts_into_transitions=True)
    assert isinstance(sac, SAC)
