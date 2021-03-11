from maze.core.agent.dummy_cartpole_policy import DummyCartPolePolicy
from maze.core.rollout.sequential_rollout_runner import SequentialRolloutRunner
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv

# Instantiate an example environment and agent
env = GymMazeEnv("CartPole-v0")
agent = DummyCartPolePolicy()

# Run a sequential rollout with rendering
# (including an example wrapper the environment will be wrapped in)
sequential = SequentialRolloutRunner(
    n_episodes=10,
    max_episode_steps=100,
    record_trajectory=True,
    record_event_logs=True,
    render=True)
sequential.run_with(
    env=env,
    wrappers={"maze.core.wrappers.reward_scaling_wrapper.RewardScalingWrapper": {"scale": 0.1}},
    agent=agent)
