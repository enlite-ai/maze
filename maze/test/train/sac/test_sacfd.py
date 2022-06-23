import pytest

from maze.test.shared_test_utils.run_maze_utils import run_maze_job


def run_sacfd(env: str, teacher_policy: str, sac_runner: str, sac_wrappers: str, sac_model: str, sac_critic: str):
    """Run soft actor critic from demonstrations for given config parameters.

    Runs a rollout with the given teacher_policy, then runs sacfD on the collected trajectory data.
    """
    # Heuristics rollout
    """Test the functionality of sacfd by first running a rollout and then starting sac with the computed output"""

    # Heuristics rollout
    rollout_config = dict(configuration="test",
                          env=env,
                          policy=teacher_policy,
                          runner="sequential")
    rollout_config['runner.n_episodes'] = 10
    rollout_config['runner.max_episode_steps'] = 10
    rollout_config["runner.record_trajectory"] = True
    run_maze_job(rollout_config, config_module="maze.conf", config_name="conf_rollout")

    # Behavioral cloning on top of the heuristic rollout trajectories
    train_config = dict(configuration="test", env=env, wrappers=sac_wrappers,
                        model=sac_model, algorithm="sacfd", runner=sac_runner, critic=sac_critic)
    run_maze_job(train_config, config_module="maze.conf", config_name="conf_train")


@pytest.mark.parametrize("runner", ["dev", "local"])
def test_sacfd(runner: str):
    """Tests the soft actor critic from demonstrations."""
    run_sacfd(env="gym_env", teacher_policy="random_policy",
               sac_runner=runner, sac_wrappers="vector_obs", sac_model="flatten_concat",
               sac_critic='flatten_concat_state_action')
