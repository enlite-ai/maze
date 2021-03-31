import pytest
from maze.test.shared_test_utils.run_maze_utils import run_maze_job


def run_behavioral_cloning(env: str, teacher_policy: str, bc_runner: str, bc_wrappers: str, bc_model: str):
    """Run behavioral cloning for given config parameters.

    Runs a rollout with the given teacher_policy, then runs behavioral cloning on the collected trajectory data.
    """
    # Heuristics rollout
    rollout_config = dict(configuration="test",
                          env=env,
                          policy=teacher_policy,
                          runner="sequential")
    run_maze_job(rollout_config, config_module="maze.conf", config_name="conf_rollout")

    # Behavioral cloning on top of the heuristic rollout trajectories
    train_config = dict(configuration="test", env=env, wrappers=bc_wrappers,
                        model=bc_model, algorithm="bc", runner=bc_runner)
    run_maze_job(train_config, config_module="maze.conf", config_name="conf_train")

    # Note: The log might output statistics multiple times -- this is caused by stats log writers being
    #       registered repeatedly in each maze_run method above (does not happen in normal scenario)


@pytest.mark.parametrize("runner", ["dev", "local"])
def test_behavioral_cloning(runner: str):
    """Rolls out a heuristic policy on Cutting 2D env and collects trajectories, then runs
    behavioral cloning on the collected trajectory data."""
    run_behavioral_cloning(env="gym_env", teacher_policy="random_policy",
                           bc_runner=runner, bc_wrappers="vector_obs", bc_model="flatten_concat")
