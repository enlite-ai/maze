"""
Tests for Supervisor run_context.
"""
import copy
from typing import Tuple

import gym
import pytest
from maze.train.trainers.ppo.ppo_trainer import PPO
from torch import nn

from maze.api import run_context
from maze.api.utils import RunMode
from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.agent.torch_state_critic import TorchSharedStateCritic, TorchStepStateCritic
from maze.core.env.maze_env import MazeEnv
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.maze_gym_env_wrapper import GymCoreEnv, GymMazeEnv
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.blocks.general.torch_model_block import TorchModelBlock
from maze.perception.models.built_in.flatten_concat import FlattenConcatPolicyNet, FlattenConcatStateValueNet
from maze.perception.models.critics import SharedStateCriticComposer
from maze.perception.models.custom_model_composer import CustomModelComposer
from maze.perception.models.policies import ProbabilisticPolicyComposer
from maze.train.trainers.a2c.a2c_algorithm_config import A2CAlgorithmConfig


def _get_cartpole_setup_components() -> Tuple[
    CustomModelComposer, ProbabilisticPolicyComposer, SharedStateCriticComposer, TorchPolicy, TorchActorCritic
]:
    """
    Returns various instantiated components for environment CartPole-v0.
    :return: Various components cartpole setting.
    """

    env = GymMazeEnv(env=gym.make("CartPole-v0"))
    observation_space = env.observation_space
    action_space = env.action_space

    policy_net = FlattenConcatPolicyNet({'observation': (4,)}, {'action': (2,)}, hidden_units=[16], non_lin=nn.Tanh)
    maze_wrapped_policy_net = TorchModelBlock(
        in_keys='observation', out_keys='action',
        in_shapes=observation_space.spaces['observation'].shape, in_num_dims=[2],
        out_num_dims=2, net=policy_net)

    policy_networks = {0: maze_wrapped_policy_net}

    # Policy Distribution
    # ^^^^^^^^^^^^^^^^^^^
    distribution_mapper = DistributionMapper(
        action_space=action_space,
        distribution_mapper_config={})

    # Instantiating the Policy
    # ^^^^^^^^^^^^^^^^^^^^^^^^
    torch_policy = TorchPolicy(networks=policy_networks,
                               distribution_mapper=distribution_mapper,
                               device='cpu')

    policy_composer = ProbabilisticPolicyComposer(
        action_spaces_dict=env.action_spaces_dict,
        observation_spaces_dict=env.observation_spaces_dict,
        distribution_mapper=distribution_mapper,
        networks=[{
            '_target_': 'maze.perception.models.built_in.flatten_concat.FlattenConcatPolicyNet',
            'non_lin': 'torch.nn.Tanh',
            'hidden_units': [222, 222]
        }],
        substeps_with_separate_agent_nets=[],
        agent_counts_dict={0: 1}
    )

    # Value Function Setup
    # --------------------

    # Value Network
    # ^^^^^^^^^^^^^
    value_net = FlattenConcatStateValueNet({'observation': (4,)}, hidden_units=[16], non_lin=nn.Tanh)
    maze_wrapped_value_net = TorchModelBlock(
        in_keys='observation', out_keys='value',
        in_shapes=observation_space.spaces['observation'].shape,
        in_num_dims=[2],
        out_num_dims=2,
        net=value_net
    )

    value_networks = {0: maze_wrapped_value_net}

    # Instantiate the Value Function
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    torch_critic = TorchSharedStateCritic(
        networks=value_networks, num_policies=1, device='cpu', stack_observations=True
    )

    # Critic composer.
    critic_composer = SharedStateCriticComposer(
        observation_spaces_dict=env.observation_spaces_dict,
        agent_counts_dict={0: 1},
        networks=value_networks,
        stack_observations=True
    )

    # Initializing the ActorCritic Model.
    # -----------------------------------
    actor_critic_model = TorchActorCritic(policy=torch_policy, critic=torch_critic, device='cpu')

    model_composer = CustomModelComposer(
        action_spaces_dict=env.action_spaces_dict,
        observation_spaces_dict=env.observation_spaces_dict,
        distribution_mapper_config={},
        policy=policy_composer,
        critic=None,
        agent_counts_dict={0: 1}
    )

    return model_composer, policy_composer, critic_composer, torch_policy, actor_critic_model


@pytest.mark.parametrize(
    "alg,runner",
    [
        ("a2c", "dev"), ("a2c", "local"),
        ("impala", "dev"), ("impala", "local"),
        ("ppo", "dev"), ("ppo", "local"),
        ("es", "dev"), ("es", "local")
    ]
)
def test_standalone_training(alg: str, runner: str) -> None:
    """
    Tests whether instantation and standalone training runs with all supported algorithms.
    :param alg: Algorithm to train with.
    :param runner: Runner configuration module name.
    """

    overrides = {
        "env.name": "CartPole-v0", "runner.normalization_samples": "1"
    }

    run_context.RunContext(
        algorithm=alg, overrides=overrides, silent=True, runner=runner, configuration="test"
    ).train(n_epochs=1)


def test_overrides() -> None:
    """
    Tests setting of values via overrides dictionary.
    """

    _, policy_composer, _, _, _ = _get_cartpole_setup_components()
    gym_env_name = "CartPole-v0"

    rc = run_context.RunContext(
        algorithm="a2c",
        overrides={
            "env.name": gym_env_name,
            "model.policy": policy_composer,
            "runner.normalization_samples": 1,
            "runner.concurrency": 1
        },
        silent=True
    )
    rc.train(n_epochs=1)

    train_network = rc._runners[run_context.RunMode.TRAINING]._trainer.model.policy.networks[0]
    env = rc._runners[run_context.RunMode.TRAINING].env_factory()
    assert isinstance(env.core_env, GymCoreEnv)
    assert env.core_env.env.unwrapped.spec.id == gym_env_name
    assert isinstance(train_network, FlattenConcatPolicyNet)
    assert train_network.hidden_units == [222, 222]


def test_multiple_runs() -> None:
    """
    Tests behaviour with execution of multiple subsequent runs.
    """

    rc = run_context.RunContext(
        silent=True,
        overrides={
            "runner.normalization_samples": 1,
            "runner.n_eval_rollouts": 1,
            "runner.shared_noise_table_size": 10
        }
    )
    rc.train(n_epochs=1)
    rc.train(n_epochs=1)


def test_template_model_composer() -> None:
    """
    Tests behaviour with TemplateModelComposer.
    """

    default_overrides = {
        "runner.normalization_samples": 1,
        "runner.n_eval_rollouts": 1,
        "runner.shared_noise_table_size": 10
    }

    run_context.RunContext(silent=True, model="vector_obs", overrides=default_overrides).train(1)

    # Plug in invalid policy.
    with pytest.raises(TypeError):
        run_context.RunContext(
            silent=True, model="vector_obs", overrides=default_overrides, policy="random_policy"
        ).train(1)

    # Specify valid policy directly.
    run_context.RunContext(
        silent=True,
        model="vector_obs",
        overrides=default_overrides,
        policy={'_target_': 'maze.perception.models.policies.ProbabilisticPolicyComposer'}
    ).train(1)

    # Specify valid policy via overrides.
    run_context.RunContext(
        overrides={
            **default_overrides,
            'model.policy._target_': 'maze.perception.models.policies.ProbabilisticPolicyComposer'
        }
    ).train(1)

    # Specify invalid policy target.
    with pytest.raises(ValueError):
        run_context.RunContext(
            silent=True,
            model="vector_obs",
            overrides={**default_overrides, 'model.policy._target_': 'x'}
        ).train(1)


def test_manual_rollout() -> None:
    """
    Test manual rollout via control loop.
    """

    env_factory = lambda: GymMazeEnv('CartPole-v0')
    rc = run_context.RunContext(env=lambda: env_factory(), silent=True)
    rc.train(n_epochs=1)

    env = env_factory()
    obs = env.reset()
    for i in range(2):
        action = rc.policy.compute_action(obs)
        action = rc.compute_action(obs)
        obs, rewards, dones, info = env.step(action)


def test_predefined_runner() -> None:
    """
    Test specification with pre-defined runner.
    """

    rc = run_context.RunContext(silent=True)
    rc = run_context.RunContext(silent=True, runner=rc._runners[run_context.RunMode.TRAINING])
    rc.train(n_epochs=1)


def test_inconsistency_identification_type_1() -> None:
    """
    Tests identification of inconsistency due to specified elements being incompatible with the run mode.
    """

    with pytest.raises(run_context.InvalidSpecificationError):
        run_context.RunContext(runner="parallel").train(1)


def test_inconsistency_identification_type_2() -> None:
    """
    Tests identification of inconsistency due to codependent components.
    """

    gym_env_name = "CartPole-v0"
    es_dev_runner_config = {
        'state_dict_dump_file': 'state_dict.pt',
        'spaces_config_dump_file': 'spaces_config.pkl',
        'normalization_samples': 1,
        '_target_': 'maze.train.trainers.es.ESDevRunner',
        'n_eval_rollouts': 1,
        'shared_noise_table_size': 10
    }
    a2c_dev_runner_config = {
        'state_dict_dump_file': 'state_dict.pt',
        'spaces_config_dump_file': 'spaces_config.pkl',
        'normalization_samples': 1,
        '_target_': 'maze.train.trainers.common.actor_critic.actor_critic_runners.ACDevRunner',
        "trainer_class": "maze.train.trainers.a2c.a2c_trainer.A2C",
        'concurrency': 1,
        "initial_state_dict": None
    }
    invalid_a2c_dev_runner_config = copy.deepcopy(a2c_dev_runner_config)
    invalid_a2c_dev_runner_config["trainer_class"] = "maze.train.trainers.es.es_trainer.ESTrainer"
    a2c_alg_config = A2CAlgorithmConfig(
        n_epochs=1,
        epoch_length=25,
        deterministic_eval=False,
        eval_repeats=2,
        patience=15,
        critic_burn_in_epochs=0,
        n_rollout_steps=100,
        lr=0.0005,
        gamma=0.98,
        gae_lambda=1.0,
        policy_loss_coef=1.0,
        value_loss_coef=0.5,
        entropy_coef=0.00025,
        max_grad_norm=0.0,
        device='cpu'
    )
    default_overrides = {"env.name": gym_env_name}

    with pytest.raises(run_context.InvalidSpecificationError):
        run_context.RunContext(
            algorithm="a2c",
            runner=es_dev_runner_config,
            silent=True,
            overrides=default_overrides
        )

    with pytest.raises(run_context.InvalidSpecificationError):
        run_context.RunContext(
            algorithm=a2c_alg_config,
            runner=es_dev_runner_config,
            silent=True,
            overrides=default_overrides
        )

    with pytest.raises(run_context.InvalidSpecificationError):
        run_context.RunContext(
            algorithm="es",
            runner=a2c_dev_runner_config,
            silent=True,
            overrides=default_overrides
        )

    with pytest.raises(run_context.InvalidSpecificationError):
        run_context.RunContext(
            algorithm="a2c",
            runner=invalid_a2c_dev_runner_config,
            silent=True,
            overrides=default_overrides
        )

    rc = run_context.RunContext(
        algorithm="es",
        runner=es_dev_runner_config,
        silent=True,
        overrides=default_overrides
    )
    rc.train(1)

    rc = run_context.RunContext(
        algorithm="a2c",
        runner=a2c_dev_runner_config,
        silent=True,
        overrides=default_overrides
    )
    rc.train(1)


def test_inconsistency_identification_type_3() -> None:
    """
    Tests identification of inconsistency due to derived config group.
    """

    es_dev_runner_config = {
        'state_dict_dump_file': 'state_dict.pt',
        'spaces_config_dump_file': 'spaces_config.pkl',
        'normalization_samples': 10000,
        '_target_': 'maze.train.trainers.es.ESDevRunner',
        'n_eval_rollouts': 10,
        'shared_noise_table_size': 100000000
    }
    a2c_alg_config = A2CAlgorithmConfig(
        n_epochs=1,
        epoch_length=25,
        deterministic_eval=False,
        eval_repeats=2,
        patience=15,
        critic_burn_in_epochs=0,
        n_rollout_steps=100,
        lr=0.0005,
        gamma=0.98,
        gae_lambda=1.0,
        policy_loss_coef=1.0,
        value_loss_coef=0.5,
        entropy_coef=0.00025,
        max_grad_norm=0.0,
        device='cpu'
    )
    default_overrides = {"runner.normalization_samples": 1, "runner.concurrency": 1}

    rc = run_context.RunContext(
        algorithm=a2c_alg_config,
        env=lambda: GymMazeEnv(env="CartPole-v0"),
        silent=True,
        runner="dev",
        overrides=default_overrides
    )
    rc.train(1)

    run_context.RunContext(
        env=lambda: GymMazeEnv(env="CartPole-v0"),
        runner=es_dev_runner_config,
        silent=True,
        overrides=default_overrides
    )
    rc.train(1)


def test_inconsistency_identification_type_4_invalid() -> None:
    """
    Tests identification of inconsistency due to specification of super- and subcomponents.
    """

    model_composer, policy_composer, _, _, _ = _get_cartpole_setup_components()
    model_policy_target = "maze.perception.models.policies.ProbabilisticPolicyComposer"

    # With nesting level > 1. Both parent and child in overrides.
    with pytest.raises(run_context.InvalidSpecificationError):
        run_context.RunContext(
            algorithm="a2c",
            env=lambda: GymMazeEnv(env="CartPole-v0"),
            silent=True,
            overrides={
                "policy": policy_composer,
                "model.policy._target_": model_policy_target
            }
        )

    # With nesting level > 1, parent in overrides with proxy path.
    with pytest.raises(run_context.InvalidSpecificationError):
        run_context.RunContext(
            algorithm="a2c",
            env=lambda: GymMazeEnv(env="CartPole-v0"),
            silent=True,
            policy=policy_composer,
            overrides={
                "model.policy._target_": model_policy_target
            }
        )

    # With nesting level > 1, with proxy path, parent with full path.
    with pytest.raises(run_context.InvalidSpecificationError):
        run_context.RunContext(
            algorithm="a2c",
            env=lambda: GymMazeEnv(env="CartPole-v0"),
            silent=True,
            overrides={
                "policy": policy_composer,
                "policy._target_": model_policy_target
            }
        )

    # With nesting level > 1, both with proxy path.
    with pytest.raises(run_context.InvalidSpecificationError):
        run_context.RunContext(
            algorithm="a2c",
            env=lambda: GymMazeEnv(env="CartPole-v0"),
            silent=True,
            overrides={
                "policy": policy_composer,
                "policy._target_": model_policy_target
            }
        )

    # With nesting level > 1, parent as explicit argument.
    with pytest.raises(run_context.InvalidSpecificationError):
        run_context.RunContext(
            algorithm="a2c",
            env=lambda: GymMazeEnv(env="CartPole-v0"),
            silent=True,
            policy=policy_composer,
            overrides={
                "model.policy._target_": model_policy_target
            }
        )

    # With override referencing explicit argument.
    with pytest.raises(run_context.InvalidSpecificationError):
        run_context.RunContext(
            algorithm="a2c",
            env=lambda: GymMazeEnv(env="CartPole-v0"),
            silent=True,
            model=model_composer,
            overrides={"model.policy": policy_composer}
        )

    # With override referencing explicit argument via proxy.
    with pytest.raises(run_context.InvalidSpecificationError):
        run_context.RunContext(
            algorithm="a2c",
            env=lambda: GymMazeEnv(env="CartPole-v0"),
            silent=True,
            model=model_composer,
            overrides={"policy": policy_composer}
        )

    # With explicit argument referencing other explicit argument.
    with pytest.raises(run_context.InvalidSpecificationError):
        run_context.RunContext(
            algorithm="a2c",
            env=lambda: GymMazeEnv(env="CartPole-v0"),
            silent=True,
            model=model_composer,
            policy=policy_composer
        )


def test_inconsistency_identification_type_4_valid() -> None:
    """
    Tests identification of inconsistency due to specification of super- and subcomponents.
    """

    _, policy_composer, _, _, _ = _get_cartpole_setup_components()
    model_policy_target = "maze.perception.models.policies.ProbabilisticPolicyComposer"
    model_dictconfig = {
        '_target_': 'maze.perception.models.custom_model_composer.CustomModelComposer',
        'distribution_mapper_config': [{
            'action_space': 'gym.spaces.Box',
            'distribution': 'maze.distributions.squashed_gaussian.SquashedGaussianProbabilityDistribution'
        }],
        'policy': {
            '_target_': 'maze.perception.models.policies.ProbabilisticPolicyComposer',
            'networks': [{
                '_target_': 'maze.perception.models.built_in.flatten_concat.FlattenConcatPolicyNet',
                'non_lin': 'torch.nn.Tanh',
                'hidden_units': [222, 222]
            }],
            "substeps_with_separate_agent_nets": [],
            "agent_counts_dict": {0: 1}
        },
        'critic': {
            '_target_': 'maze.perception.models.critics.StateCriticComposer',
            'networks': [{
                '_target_': 'maze.perception.models.built_in.flatten_concat.FlattenConcatStateValueNet',
                'non_lin': 'torch.nn.Tanh',
                'hidden_units': [256, 256]
            }]
        }
    }
    default_overrides = {"runner.concurrency": 1}

    # With DictConfig parent (legal).
    rc = run_context.RunContext(
        algorithm="a2c",
        env=lambda: GymMazeEnv(env="CartPole-v0"),
        silent=True,
        model=model_dictconfig,
        policy=policy_composer,
        runner="dev",
        overrides=default_overrides,
        configuration="test"
    )
    rc.train(1)
    assert rc._runners[run_context.RunMode.TRAINING]._model_composer.policy.networks[0].hidden_units == [222, 222]

    # With config module name parent.
    rc = run_context.RunContext(
        algorithm="a2c",
        env=lambda: GymMazeEnv(env="CartPole-v0"),
        silent=True,
        model="flatten_concat",
        policy=policy_composer,
        runner="dev",
        overrides=default_overrides,
        configuration="test"
    )
    rc.train(1)
    assert rc._runners[run_context.RunMode.TRAINING]._model_composer.policy.networks[0].hidden_units == [222, 222]

    # With config module name parent and aliased child override.
    rc = run_context.RunContext(
        algorithm="a2c",
        env=lambda: GymMazeEnv(env="CartPole-v0"),
        silent=True,
        policy=model_dictconfig["policy"],
        runner="dev",
        overrides={"policy._target_": model_policy_target, **default_overrides},
        configuration="test"
    )
    rc.train(1)
    assert rc._runners[run_context.RunMode.TRAINING]._model_composer.policy.networks[0].hidden_units == [222, 222]

    # With config module name parent and non-aliased child override.
    rc = run_context.RunContext(
        algorithm="a2c",
        env=lambda: GymMazeEnv(env="CartPole-v0"),
        silent=True,
        policy=model_dictconfig["policy"],
        runner="dev",
        overrides={"model.policy._target_": model_policy_target},
        configuration="test"
    )
    rc.train(1)
    assert rc._runners[run_context.RunMode.TRAINING]._model_composer.policy.networks[0].hidden_units == [222, 222]


def test_env_type():
    """
    Tests whether environment is correctly wrapped.
    """

    rc = run_context.RunContext(
        silent=True,
        overrides={"runner.normalization_samples": 1, "runner.shared_noise_table_size": 10}
    )
    rc.train(1)
    env = rc._runners[run_context.RunMode.TRAINING].env_factory()

    assert isinstance(env, MazeEnv)
    assert isinstance(env, LogStatsWrapper)


def test_experiment():
    """
    Tests whether experiments are correctly loaded.
    """

    rc = run_context.RunContext(
        env=lambda: GymMazeEnv('CartPole-v0'),
        silent=True,
        overrides={"runner.normalization_samples": 1, "runner.concurrency": 1},
        experiment="cartpole_ppo_wrappers"
    )
    rc.train(1)

    assert isinstance(rc._runners[RunMode.TRAINING]._trainer, PPO)
    assert rc._runners[RunMode.TRAINING]._cfg.algorithm.lr == 0.0001


def test_autoresolving_proxy_attribute():
    """
    Tests auto-resolving proxy attributes like critic (see for :py:class:`maze.api.utils._ATTRIBUTE_PROXIES` for more
    info).
    """

    cartpole_env_factory = lambda: GymMazeEnv(env=gym.make("CartPole-v0"))

    _, _, critic_composer, _, _ = _get_cartpole_setup_components()
    alg_config = A2CAlgorithmConfig(
        n_epochs=1,
        epoch_length=25,
        deterministic_eval=False,
        eval_repeats=2,
        patience=15,
        critic_burn_in_epochs=0,
        n_rollout_steps=100,
        lr=0.0005,
        gamma=0.98,
        gae_lambda=1.0,
        policy_loss_coef=1.0,
        value_loss_coef=0.5,
        entropy_coef=0.00025,
        max_grad_norm=0.0,
        device='cpu'
    )
    default_overrides = {"runner.normalization_samples": 1, "runner.concurrency": 1}

    rc = run_context.RunContext(
        env=cartpole_env_factory,
        silent=True,
        algorithm=alg_config,
        critic=critic_composer,
        runner="dev",
        overrides=default_overrides
    )
    rc.train(n_epochs=1)
    assert isinstance(rc._runners[RunMode.TRAINING].model_composer.critic, TorchSharedStateCritic)

    rc = run_context.RunContext(
        env=cartpole_env_factory,
        silent=True,
        algorithm=alg_config,
        critic="template_state",
        runner="dev",
        overrides=default_overrides
    )
    rc.train(n_epochs=1)
    assert isinstance(rc._runners[RunMode.TRAINING].model_composer.critic, TorchStepStateCritic)


def test_evaluation():
    """
    Tests evaluation.
    """

    rc = run_context.RunContext(
        env=lambda: GymMazeEnv(env=gym.make("CartPole-v0")),
        silent=True,
        overrides={"runner.normalization_samples": 1, "runner.shared_noise_table_size": 10}
    )
    rc.train(1)

    # Evaluate sequentially.
    rc.evaluate(5, 5)

    # Evaluate in parallel.
    rc.evaluate(5, 5, parallel=True)
