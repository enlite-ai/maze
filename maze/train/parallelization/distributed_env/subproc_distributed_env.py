import multiprocessing
import pickle
from typing import Callable, List, Iterable, Any, Tuple, Dict, Optional, Union

import gym
import numpy as np
import cloudpickle

from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats import LogStatsLevel, LogStatsAggregator, LogStatsValue, get_stats_logger, \
    LogStatsConsumer, LogStats
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.train.parallelization.distributed_env.distributed_env import BaseDistributedEnv
from maze.train.parallelization.observation_aggregator import DictObservationAggregator


class SinkHoleConsumer(LogStatsConsumer):
    """Sink hole statistics consumer. Discards all statistics on receive."""

    def receive(self, stat: LogStats):
        """Do not keep the received statistics."""
        pass


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env: Union[StructuredEnv, StructuredEnvSpacesMixin] = env_fn_wrapper.var()

    # enable collection of logging statistics
    if not isinstance(env, LogStatsWrapper):
        env = LogStatsWrapper.wrap(env)

    # discard epoch-level statistics (as stats are shipped to the main process after each episode)
    sink_hole_consumer = SinkHoleConsumer()
    env.stats_map[LogStatsLevel.EPOCH] = sink_hole_consumer
    env.stats_map[LogStatsLevel.EPISODE].consumers = [sink_hole_consumer]

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

                remote.send((observation, reward, env_done, info, actor_done, actor_id, episode_stats))
            elif cmd == 'seed':
                env.seed(data)
            elif cmd == 'reset':
                observation = env.reset()
                remote.send((observation, env.get_stats(LogStatsLevel.EPISODE).last_stats))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_spaces_dict, env.action_spaces_dict))
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


class SubprocStructuredDistributedEnv(BaseDistributedEnv, StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv):
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
                 env_factories: List[Callable[[], StructuredEnv]],
                 logging_prefix: Optional[str] = None,
                 start_method: str = None):
        super().__init__(num_envs=len(env_factories))

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
        self._observation_spaces_dict, self._action_spaces_dict = self.remotes[0].recv()

        # initialize observation aggregation
        self.obs_aggregator = DictObservationAggregator()

        # keep track of registered logging statistics consumers
        self.epoch_stats = LogStatsAggregator(LogStatsLevel.EPOCH)

        if logging_prefix is not None:
            self.epoch_stats.register_consumer(get_stats_logger(logging_prefix))

    def step(self, actions: Iterable[Any]) -> \
            Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Iterable[Dict[Any, Any]]]:
        """Step the environments with the given actions.

        :param actions: the list of actions for the respective envs.
        :return: observations, rewards, dones, information-dicts all in env-aggregated form.
        """
        self._step_async(actions)
        return self._step_wait()

    def reset(self) -> Dict[str, np.ndarray]:
        """BaseDistributedEnv implementation"""
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, episode_stats = zip(*results)

        # collect episode statistics
        for stat in episode_stats:
            if stat is not None:
                self.epoch_stats.receive(stat)

        # aggregate list observations
        self.obs_aggregator.reset(obs)
        return self.obs_aggregator.aggregate()

    def seed(self, seed=None) -> None:
        """BaseDistributedEnv implementation"""
        for idx, remote in enumerate(self.remotes):
            remote.send(('seed', seed + idx))

    def close(self) -> None:
        """BaseDistributedEnv implementation"""
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
        obs, rews, env_dones, infos, actor_dones, actor_ids, episode_stats = zip(*results)

        self._actor_dones = np.stack(actor_dones)
        self._actor_ids = actor_ids

        # collect episode statistics
        for stat in episode_stats:
            if stat is not None:
                self.epoch_stats.receive(stat)

        self.obs_aggregator.reset(obs)
        return self.obs_aggregator.aggregate(), np.stack(rews), np.stack(env_dones), infos

    @override(LogStatsEnv)
    def get_stats(self, level: LogStatsLevel) -> LogStatsAggregator:
        """Returns the aggregator of the individual episode statistics emitted by the parallel envs.

        :param level: Must be set to `LogStatsLevel.EPOCH`, step or episode statistics are not propagated
        """

        # support only epoch statistics
        assert level == LogStatsLevel.EPOCH

        return self.epoch_stats

    @override(LogStatsEnv)
    def write_epoch_stats(self):
        """Trigger the epoch statistics generation."""
        self.epoch_stats.reduce()

    @override(LogStatsEnv)
    def get_stats_value(self,
                        event: Callable,
                        level: LogStatsLevel,
                        name: Optional[str] = None) -> LogStatsValue:
        """Obtain a single value from the epoch statistics dict.

        :param event: The event interface method of the value in question.
        :param name: The *output_name* of the statistics in case it has been specified in
                     :func:`maze.core.log_stats.event_decorators.define_epoch_stats`
        :param level: Must be set to `LogStatsLevel.EPOCH`, step or episode statistics are not propagated.
        """
        assert level == LogStatsLevel.EPOCH

        return self.epoch_stats.last_stats[(event, name, None)]

    @override(StructuredEnv)
    def actor_id(self) -> List[Tuple[Union[str, int], int]]:
        """Vectorized version of StructuredEnv.actor_id"""
        return self._actor_ids

    @override(StructuredEnv)
    def is_actor_done(self) -> np.ndarray:
        """Vectorized version of StructuredEnv.is_done"""
        return self._actor_dones

    @property
    def action_spaces_dict(self) -> Dict[Union[int, str], gym.Space]:
        """Return the action space of one of the distributed envs."""
        return self._action_spaces_dict

    @property
    def observation_spaces_dict(self) -> Dict[Union[int, str], gym.Space]:
        """Return the observation space of one of the distributed envs."""
        return self._observation_spaces_dict

    @property
    @override(StructuredEnvSpacesMixin)
    def action_space(self) -> gym.spaces.Space:
        """implementation of :class:`~maze.core.env.structured_env_spaces_mixin.StructuredEnvSpacesMixin` interface
        """
        sub_step_id = self.actor_id[0][0]
        return self.action_spaces_dict[sub_step_id]

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_space(self) -> gym.spaces.Space:
        """implementation of :class:`~maze.core.env.structured_env_spaces_mixin.StructuredEnvSpacesMixin` interface
        """
        sub_step_id = self.actor_id[0][0]
        return self.observation_space[sub_step_id]