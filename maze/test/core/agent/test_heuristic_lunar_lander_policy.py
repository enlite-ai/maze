"""Contains a unit tests for the lunar lander heuristics"""
from maze.core.agent.heuristic_lunar_lander_policy import HeuristicLunarLanderPolicy
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv


def test_heuristic_lunar_lander_policy():
    """unit tests"""
    policy = HeuristicLunarLanderPolicy()
    env = GymMazeEnv("LunarLander-v2")

    obs = env.reset()
    action = policy.compute_action(obs)
    obs, _, _, _ = env.step(action)
