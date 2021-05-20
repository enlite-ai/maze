"""
Tests for exemplary RunContext uses given in maze/docs/source/concepts_and_structure/run_context_overview.rst.
"""
import omegaconf
import pytest
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.models.policies import ProbabilisticPolicyComposer
from maze.train.trainers.a2c.a2c_algorithm_config import A2CAlgorithmConfig
from maze.api.run_context import RunContext


def _get_alg_config() -> A2CAlgorithmConfig:
    """
    Returns algorithm config used in tests.
    :return: A2CAlgorithmConfig instance.
    """

    return A2CAlgorithmConfig(
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


def test_examples_part1():
    """
    Tests snippets in maze/docs/source/concepts_and_structure/run_context_overview.rst.
    Adds some performance-specific configuration that should not influence snippets' functionality.
    Split for runtime reasons.
    """

    a2c_overrides = {"runner.concurrency": 1}
    es_overrides = {"algorithm.n_epochs": 1, "algorithm.n_rollouts_per_update": 1}
    env_factory = lambda: GymMazeEnv('CartPole-v0')

    # ------------------------------------------------------------------

    rc = RunContext(
        algorithm="a2c",
        overrides={"env.name": "CartPole-v0", **a2c_overrides},
        model="vector_obs",
        critic="template_state",
        runner="dev",
        configuration="test"
    )
    rc.train(n_epochs=1)

    # ------------------------------------------------------------------

    alg_config = _get_alg_config()
    rc = RunContext(
        algorithm=alg_config,
        overrides={"env.name": "CartPole-v0", **a2c_overrides},
        model="vector_obs",
        critic="template_state",
        runner="dev",
        configuration="test"
    )
    rc.train(n_epochs=1)

    # ------------------------------------------------------------------

    rc = RunContext(env=lambda: GymMazeEnv('CartPole-v0'), overrides=es_overrides, runner="dev", configuration="test")
    rc.train(n_epochs=1)

    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------

    env = env_factory()
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


def test_examples_part2():
    """
    Tests snippets in maze/docs/source/concepts_and_structure/run_context_overview.rst.
    Adds some performance-specific configuration that should not influence snippets' functionality.
    Split for runtime reasons.
    """

    a2c_overrides = {"runner.concurrency": 1}
    es_overrides = {"algorithm.n_epochs": 1, "algorithm.n_rollouts_per_update": 1}
    env_factory = lambda: GymMazeEnv('CartPole-v0')
    alg_config = _get_alg_config()

    # ------------------------------------------------------------------

    rc = RunContext(algorithm=alg_config, runner="dev", overrides=a2c_overrides, configuration="test")
    rc.train(n_epochs=1)

    # ------------------------------------------------------------------

    with pytest.raises(omegaconf.errors.ConfigAttributeError):
        rc = RunContext(overrides={"algorithm": alg_config}, runner="dev")
        rc.train(n_epochs=1)

    # ------------------------------------------------------------------

    rc = RunContext(env=lambda: env_factory(), overrides=es_overrides, runner="dev", configuration="test")
    rc.train(n_epochs=1)

    # Run trained policy.
    env = env_factory()
    obs = env.reset()
    for i in range(10):
        action = rc.compute_action(obs)
        obs, rewards, dones, info = env.step(action)

    # ------------------------------------------------------------------

    rc = RunContext(env=lambda: GymMazeEnv('CartPole-v0'), overrides=es_overrides, runner="dev", configuration="test")
    rc.train()

    evaluator = RolloutEvaluator(
        # Environment has to be have statistics logging capabilities for RolloutEvaluator.
        eval_env=LogStatsWrapper.wrap(env_factory(), logging_prefix="eval"),
        n_episodes=1,
        model_selection=None
    )
    evaluator.evaluate(rc.policy)
