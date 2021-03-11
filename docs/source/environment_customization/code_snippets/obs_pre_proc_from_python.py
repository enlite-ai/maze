"""Contains an example showing how to use observation pre-processing directly from python."""
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.core.wrappers.observation_preprocessing.preprocessing_wrapper import PreProcessingWrapper

# this is the pre-processor config as a python dict
config = {
    "pre_processor_mapping": [
        {"observation": "observation",
         "_target_": "maze.preprocessors.Rgb2GrayPreProcessor",
         "keep_original": False,
         "config": {"rgb_dim": -1}},
    ]
}

# instantiate a maze environment
env = GymMazeEnv("CarRacing-v0")

# wrap the environment for observation pre-processing
env = PreProcessingWrapper.wrap(env, pre_processor_mapping=config["pre_processor_mapping"])

# after this step the training env yields pre-processed observations
pre_processed_obs = env.reset()
