import gymnasium as gym

from maze.core.agent.random_policy import RandomPolicy
from maze.core.agent_deployment.agent_deployment import AgentDeployment
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv

env = GymMazeEnv("CartPole-v1", render_mode=None)
policy = RandomPolicy(action_spaces_dict=env.action_spaces_dict)

agent_deployment = AgentDeployment(
    policy=policy,
    env=env
)

# Simulate an external production environment that does not use Maze
external_env = gym.make("CartPole-v1", render_mode=None)

maze_state, _ = external_env.reset()
reward, terminated, truncated, info = 0, False, False,{}

for i in range(10):
    # Query the agent deployment for maze action, then step the environment with it
    maze_action = agent_deployment.act(maze_state, reward, terminated, truncated, info)
    maze_state, reward, terminated, truncated, info = external_env.step(maze_action)

agent_deployment.close(maze_state, reward, terminated, truncated, info)
