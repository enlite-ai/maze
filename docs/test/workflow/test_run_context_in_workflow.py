"""
Tests for RunContext examples given in maze/docs/source/workflow/.
"""
from maze.api.run_context import RunContext


def test_experimenting_basics():
    """
    Tests for RunContext basic usage examples given in maze/docs/source/workflow/experimenting.rst.
    """

    a2c_overrides = {"runner.concurrency": 1}

    rc = RunContext(
        algorithm="ppo",
        overrides={
            "env.name": "CartPole-v0",
            "algorithm.lr": 0.0001,
            **a2c_overrides
        },
        configuration="test",
        silent=True
    )
    rc.train(n_epochs=1)

    rc = RunContext(
        experiment="cartpole_ppo_wrappers",
        overrides=a2c_overrides,
        configuration="test",
        silent=True
    )
    rc.train(n_epochs=1)


def test_experimenting_gridsearch():
    """
    Tests for RunContext gridsearch examples given in maze/docs/source/workflow/experimenting.rst.
    """

    a2c_overrides = {"runner.concurrency": 1}

    rc = RunContext(
        algorithm="ppo",
        overrides={
            "env.name": "CartPole-v0",
            "algorithm.n_epochs": 5,
            "algorithm.lr": [0.0001, 0.0005, 0.001],
            **a2c_overrides
        },
        configuration="test",
        multirun=True,
        silent=True
    )
    rc.train(n_epochs=1)

    rc = RunContext(
        algorithm="ppo",
        overrides={
            "env.name": "CartPole-v0",
            "algorithm.n_epochs": 5,
            "algorithm.lr": [0.0001, 0.0005, 0.001],
            **a2c_overrides
        },
        experiment="grid_search",
        configuration="test",
        multirun=True,
        silent=True
    )
    rc.train(n_epochs=1)
