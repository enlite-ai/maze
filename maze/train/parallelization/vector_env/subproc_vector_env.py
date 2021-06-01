import multiprocessing
from typing import Callable, List, Iterable, Any, Tuple, Dict, Optional

import cloudpickle
import matplotlib
import numpy as np

from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType
from maze.core.log_stats.log_stats import LogStatsLevel
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.train.parallelization.vector_env.structured_vector_env import StructuredVectorEnv
from maze.train.parallelization.vector_env.vector_env import VectorEnv
from maze.train.parallelization.vector_env.vector_env_utils import disable_epoch_level_stats
from maze.train.utils.train_utils import stack_numpy_dict_list, unstack_numpy_list_dict


def _worker(remote, parent_remote, env_fn_wrapper):
    # switch to non-interactive matplotlib backend
    matplotlib.use('Agg')

    parent_remote.close()
    env: MazeEnv = env_fn_wrapper.var()

    # enable collection of logging statistics
    if not isinstance(env, LogStatsWrapper):
        env = LogStatsWrapper.wrap(env)

    # discard epoch-level statistics (as stats are shipped to the main process after each episode)
    env = disable_epoch_level_stats(env)

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, env_done, info = env.step(data)
                actor_done = env.is_actor_done()
                actor_id = env.actor_id()

                episode_stats = None
                if env_done:
                    # save final observation where user can get it, then reset
                    info['terminal_observation'] = observation
                    observation = env.reset()
                    # collect episode stats after the reset
                    episode_stats = env.get_stats(LogStatsLevel.EPISODE).last_stats

                remote.send((observation, reward, env_done, info, actor_done, actor_id, episode_stats,
                             env.get_env_time()))
            elif cmd == 'seed':
                env.seed(data)
            elif cmd == 'reset':
                observation = env.reset()
                actor_done = env.is_actor_done()
                actor_id = env.actor_id()
                remote.send((observation, actor_done, actor_id, env.get_stats(LogStatsLevel.EPISODE).last_stats,
                             env.get_env_time()))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_spaces_dict, env.action_spaces_dict, env.agent_counts_dict))
            elif cmd == 'get_actor_rewards':
                remote.send(env.get_actor_rewards())
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError
        except EOFError:
            break


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle).

    :param var: (Any) the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var):
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        self.var = cloudpickle.loads(obs)


class SubprocVectorEnv(StructuredVectorEnv):
    """
    Creates a multiprocess wrapper for multiple environments, distributing each environment to its own
    process. This allows a significant speed up when the environment is computationally complex.
    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::
        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_factories: A list of functions that will create the environments
        (each callable returns a `MultiStepEnvironment` instance when called).
    :param start_method: Method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self,
                 env_factories: List[Callable[[], MazeEnv]],
                 logging_prefix: Optional[str] = None,
                 start_method: str = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_factories)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_factories):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_spaces_dict, action_spaces_dict, agent_counts_dict = self.remotes[0].recv()

        super().__init__(
            n_envs=n_envs,
            action_spaces_dict=action_spaces_dict,
            observation_spaces_dict=observation_spaces_dict,
            agent_counts_dict=agent_counts_dict,
            logging_prefix=logging_prefix
        )

    @override(StructuredVectorEnv)
    def get_actor_rewards(self) -> Optional[np.ndarray]:
        """Stack actor rewards from encapsulated environments."""
        for remote in self.remotes:
            remote.send(('get_actor_rewards', None))
        rewards = [remote.recv() for remote in self.remotes]

        # Return none if rewards are not available
        if rewards[0] is None:
            return None

        rewards = np.stack(rewards, axis=1).astype(np.float32)
        return rewards

    def step(self, actions: ActionType) -> Tuple[ObservationType, np.ndarray, np.ndarray, Iterable[Dict[Any, Any]]]:
        """Step the environments with the given actions.

        :param actions: the list of actions for the respective envs.
        :return: observations, rewards, dones, information-dicts all in env-aggregated form.
        """
        actions = unstack_numpy_list_dict(actions)
        self._step_async(actions)
        return self._step_wait()

    def reset(self) -> Dict[str, np.ndarray]:
        """VectorEnv implementation"""
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, actor_dones, actor_ids, episode_stats, env_times = zip(*results)

        self._env_times = np.stack(env_times)
        self._actor_dones = np.stack(actor_dones)
        self._actor_ids = actor_ids

        # collect episode statistics
        for stat in episode_stats:
            if stat is not None:
                self.epoch_stats.receive(stat)

        return stack_numpy_dict_list(obs)

    @override(VectorEnv)
    def seed(self, seeds: List[Any]) -> None:
        """VectorEnv implementation"""
        for (remote, seed) in zip(self.remotes, seeds):
            remote.send(('seed', seed))

    def close(self) -> None:
        """VectorEnv implementation"""
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def _step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def _step_wait(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Iterable[Dict[Any, Any]]]:
        """
        Wait for the step taken with step_async().

        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, env_dones, infos, actor_dones, actor_ids, episode_stats, env_times = zip(*results)

        self._env_times = np.stack(env_times)
        self._actor_dones = np.stack(actor_dones)
        self._actor_ids = actor_ids

        # collect episode statistics
        for stat in episode_stats:
            if stat is not None:
                self.epoch_stats.receive(stat)

        return stack_numpy_dict_list(obs), np.stack(rews), np.stack(env_dones), infos
