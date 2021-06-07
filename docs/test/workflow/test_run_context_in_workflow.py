"""
Tests for RunContext examples given in maze/docs/source/workflow/.
"""
from maze.api.run_context import RunContext


def test_experimenting():
    """
    Tests for RunContext examples given in maze/docs/source/workflow/experimenting.rst.
    """

    a2c_overrides = {"runner.concurrency": 1}

    rc = RunContext(
        algorithm="ppo",
        overrides={
            "env.name": "CartPole-v0",
            "algorithm.lr": 0.0001,
            **a2c_overrides
        },
        configuration="test"
    )
    rc.train(n_epochs=1)

    rc = RunContext(experiment="cartpole_ppo_wrappers", overrides=a2c_overrides, configuration="test")
    rc.train(n_epochs=1)
