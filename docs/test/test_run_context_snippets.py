"""
Tests RunContext snippets in various files.
"""
import pytest

from maze.api.run_context import RunContext
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.models.policies import ProbabilisticPolicyComposer
from maze.train.trainers.a2c.a2c_algorithm_config import A2CAlgorithmConfig
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator


def test_getting_started_maze_env_train_event_env():
    """
    Tests snippets in docs/source/getting_started/maze_env/train_event_env.rst.
    """

    # Only commands in file use cutting2d resource, which are not available in Maze' main repository.
    pass


def test_getting_started_maze_env_train_maze_env():
    """
    Tests snippets in docs/source/getting_started/maze_env/train_event_env.rst.
    """

    # Only commands in file use cutting2d resource, which are not available in Maze' main repository.
    pass


def test_workflow_training_part1():
    """
    Tests snippets in docs/source/workflow/training.rst. Split for runtime reasons.
    """

    # Default overrides for faster tests. Shouldn't change functionality.
    es_overrides = {"algorithm.n_epochs": 1, "algorithm.n_rollouts_per_update": 1}

    rc = RunContext(env="gym_env", overrides={"env.name": "CartPole-v0", **es_overrides}, configuration="test")
    rc.train(n_epochs=1)

    rc = RunContext(env="gym_env", overrides={"env.name": "LunarLander-v2", **es_overrides}, configuration="test")
    rc.train(n_epochs=1)

    rc = RunContext(
        env="gym_env",
        overrides={"env.name": "LunarLander-v2", **es_overrides},
        wrappers="vector_obs",
        model="vector_obs",
        configuration="test"
    )
    rc.train(n_epochs=1)


def test_workflow_training_part2():
    """
    Tests snippets in docs/source/workflow/training.rst. Split for runtime reasons.
    """

    # Default overrides for faster tests. Shouldn't change functionality.
    ac_overrides = {"runner.concurrency": 1}

    rc = RunContext(
        env="gym_env",
        overrides={"env.name": "LunarLander-v2", **ac_overrides},
        algorithm="ppo",
        configuration="test"
    )
    rc.train(n_epochs=1)

    rc = RunContext(
        env="gym_env",
        run_dir="outputs/experiment_dir",
        overrides={"env.name": "LunarLander-v2", "algorithm.lr": 0.0001, **ac_overrides},
        algorithm="ppo",
        configuration="test"
    )
    rc.train(n_epochs=1)


def test_trainers_maze_trainers():
    """
    Tests snippets in docs/source/trainers/maze_trainers.rst.
    Split for runtime reasons.
    """

    # Default overrides for faster tests. Shouldn't change functionality.
    ac_overrides = {"runner.concurrency": 1}
    es_overrides = {"algorithm.n_epochs": 1, "algorithm.n_rollouts_per_update": 1}

    rc = RunContext(
        algorithm="a2c",
        overrides={"env.name": "CartPole-v0", **ac_overrides},
        model="vector_obs",
        critic="template_state",
        runner="dev",
        configuration="test"
    )
    rc.train(n_epochs=1)

    rc = RunContext(
        algorithm="ppo",
        overrides={"env.name": "CartPole-v0", **ac_overrides},
        model="vector_obs",
        critic="template_state",
        runner="dev",
        configuration="test"
    )
    rc.train(n_epochs=1)

    rc = RunContext(
        algorithm="impala",
        overrides={"env.name": "CartPole-v0"},
        model="vector_obs",
        critic="template_state",
        runner="dev",
        configuration="test"
    )
    rc.train(n_epochs=1)

    rc = RunContext(
        algorithm="es",
        overrides={"env.name": "CartPole-v0", **es_overrides},
        model="vector_obs",
        runner="dev",
        configuration="test"
    )
    rc.train(n_epochs=1)


def test_concepts_and_structures_run_context_overview():
    """
    Tests snippets in docs/source/concepts_and_structure/run_context_overview.rst.
    """

    # Default overrides for faster tests. Shouldn't change functionality.
    ac_overrides = {"runner.concurrency": 1}
    es_overrides = {"algorithm.n_epochs": 1, "algorithm.n_rollouts_per_update": 1}

    # Training
    # --------

    rc = RunContext(
        algorithm="a2c",
        overrides={"env.name": "CartPole-v0", **ac_overrides},
        model="vector_obs",
        critic="template_state",
        runner="dev",
        configuration="test"
    )
    rc.train(n_epochs=1)

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

    rc = RunContext(
        algorithm=alg_config,
        overrides={"env.name": "CartPole-v0", **ac_overrides},
        model="vector_obs",
        critic="template_state",
        runner="dev",
        configuration="test"
    )
    rc.train(n_epochs=1)

    rc = RunContext(env=lambda: GymMazeEnv('CartPole-v0'), overrides=es_overrides, runner="dev", configuration="test")
    rc.train(n_epochs=1)

    policy_composer_config = {
        '_target_': 'maze.perception.models.policies.ProbabilisticPolicyComposer',
        'networks': [{
            '_target_': 'maze.perception.models.built_in.flatten_concat.FlattenConcatPolicyNet',
            'non_lin': 'torch.nn.Tanh',
            'hidden_units': [256, 256]
        }],
        "substeps_with_separate_agent_nets": [],
        "agent_counts_dict": {0: 1}
    }
    rc = RunContext(
        overrides={"model.policy": policy_composer_config, **es_overrides}, runner="dev", configuration="test"
    )
    rc.train(n_epochs=1)

    env = GymMazeEnv('CartPole-v0')
    policy_composer = ProbabilisticPolicyComposer(
        action_spaces_dict=env.action_spaces_dict,
        observation_spaces_dict=env.observation_spaces_dict,
        distribution_mapper=DistributionMapper(action_space=env.action_space, distribution_mapper_config={}),
        networks=[{
            '_target_': 'maze.perception.models.built_in.flatten_concat.FlattenConcatPolicyNet',
            'non_lin': 'torch.nn.Tanh',
            'hidden_units': [222, 222]
        }],
        substeps_with_separate_agent_nets=[],
        agent_counts_dict={0: 1}
    )
    rc = RunContext(overrides={"model.policy": policy_composer, **es_overrides}, runner="dev", configuration="test")
    rc.train(n_epochs=1)

    rc = RunContext(algorithm=alg_config, overrides=ac_overrides, runner="dev", configuration="test")
    rc.train(n_epochs=1)
    rc.train()

    # Rollout
    # -------

    obs = env.reset()
    for i in range(10):
        action = rc.compute_action(obs)
        obs, rewards, dones, info = env.step(action)

    # Evaluation
    # ----------

    env.reset()
    evaluator = RolloutEvaluator(
        # Environment has to be have statistics logging capabilities for RolloutEvaluator.
        eval_env=LogStatsWrapper.wrap(env, logging_prefix="eval"),
        n_episodes=1,
        model_selection=None
    )
    evaluator.evaluate(rc.policy)


def test_readme():
    """
    Tests snippets in readme.md.
    """

    rc = RunContext(env=lambda: GymMazeEnv('CartPole-v0'))
    rc.train(n_epochs=1)

    # Run trained policy.
    env = GymMazeEnv('CartPole-v0')
    obs = env.reset()
    for i in range(1):
        action = rc.compute_action(obs)
        obs, rewards, dones, info = env.step(action)
