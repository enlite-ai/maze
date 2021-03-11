from maze.core.env.maze_env import MazeEnv
from maze.core.utils.factory import Factory

env = Factory(MazeEnv).instantiate({
    "_target_": "maze.core.wrappers.maze_gym_env_wrapper.GymMazeEnv",
    "env": "CarRacing-v0"
})
