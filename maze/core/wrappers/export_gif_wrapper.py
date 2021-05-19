"""Contains a gif rendering export wrapper."""
import os
from datetime import datetime
from typing import Optional, Tuple, Dict, Union, Any

import gym
import imageio
import matplotlib.pyplot as plt
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.log_events.step_event_log import StepEventLog
from maze.core.wrappers.maze_gym_env_wrapper import GymCoreEnv
from maze.core.wrappers.wrapper import Wrapper


class ExportGifWrapper(Wrapper[MazeEnv]):
    """Dumps step renderings of environments as .gif files.

    Make sure to activate this only for rollouts and disable it during training (e.g. set export=False).
    Otherwise it will dump a lot off rollout GIFs to your disk.

    To convert the GIF into a mp4 video run:
    ffmpeg -r 1 -i <file-path>.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" <file-path>.mp4

    :param env: The environment to wrap.
    :param duration: Duration in seconds between consecutive image frames.
    :param export: Only if set to True a GIF is exported.
    """

    def __init__(self, env: MazeEnv, export: bool, duration: float):
        super().__init__(env)

        self._export = export
        self._duration = duration
        self._events = None
        self._writer = None

        self._is_gym_env = isinstance(self.env, gym.Env)

    @override(BaseEnv)
    def step(self, action: MazeActionType) -> Tuple[ObservationType, Any, bool, Dict[Any, Any]]:
        """Intercept ``BaseEnv.step`` and map observation."""
        observation, reward, done, info = self.env.step(action)

        if self._export:
            self._events.env_time += 1
            self._render()

        return observation, reward, done, info

    @override(BaseEnv)
    def reset(self) -> MazeActionType:
        """Intercept ``BaseEnv.reset`` and map observation."""

        # reset wrapped env
        observation = self.env.reset()

        if self._export:
            # close previous writer
            if self._writer:
                self._writer.close()

            # init new writer
            time_stamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S-%f")
            gif_name = f"rollout_{time_stamp}.gif"
            self._writer = imageio.get_writer(gif_name, mode='I', duration=self._duration)

            # render initial state
            self._events = StepEventLog(env_time=0)
            self._render()

        return observation

    def _render(self) -> None:
        """Render state to rgb image and append image stack.
        """

        # Gym style rendering
        if self._is_gym_env:
            assert isinstance(self.env.core_env, GymCoreEnv)
            img = self.env.core_env.env.render(mode="rgb_array")
        # Maze style rendering
        else:
            renderer = self.env.get_renderer()
            renderer.render(maze_state=self.env.get_maze_state(), maze_action=None, events=self._events)
            fig = plt.gcf()
            # img = np.array(fig.canvas.renderer.buffer_rgba())
            plt.savefig("tmp.png")
            img = imageio.imread("tmp.png")
            os.remove("tmp.png")
            plt.close(fig)

        # append image stack
        self._writer.append_data(img[:, :, :3])

    @override(Wrapper)
    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType],
                                         maze_action: Optional[MazeActionType], first_step_in_episode: bool) \
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        raise NotImplementedError

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'ExportGifWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        raise RuntimeError("Cloning the 'ExportGifWrapper' is not supported.")
