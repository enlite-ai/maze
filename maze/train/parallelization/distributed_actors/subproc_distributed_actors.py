"""Actors distributed across multiple processing using multiprocessing."""

import pickle
import time
import traceback
from multiprocessing.managers import BaseManager
from queue import Empty
from typing import Callable, Union, Tuple, Dict

import cloudpickle
from torch import multiprocessing

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.perception.perception_utils import convert_to_torch
from maze.train.parallelization.distributed_actors.actor import ActorAgent, ActorOutput
from maze.train.parallelization.distributed_actors.broadcasting_container import BroadcastingContainer
from maze.train.parallelization.distributed_actors.distributed_actors import BaseDistributedActors
from maze.train.trainers.impala.impala_batching import batch_outputs_time_major


class MyManager(BaseManager):
    """Basic wrapper for the BaseManager class"""
    pass


class SubprocDistributedActors(BaseDistributedActors):
    """Basic Distributed-Actors-Module using python multiprocessing.Process

    :param queue_out_of_sync_factor: this factor multiplied by the actor_batch_size gives the size of the queue.
        Therefore if the all rollouts computed can be  at most
        (queue_out_of_sync_factor + num_agents/actor_batch_size) out of sync with learner policy
    :param start_method: Method used to start the subprocesses.
       Must be one of the methods returned by multiprocessing.get_all_start_methods().
       Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self,
                 env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv]],
                 policy: TorchPolicy,
                 n_rollout_steps: int,
                 n_actors: int,
                 batch_size: int,
                 queue_out_of_sync_factor: float,
                 start_method: str):
        super().__init__(env_factory, policy, n_rollout_steps, n_actors, batch_size)

        self.queue_out_of_sync_factor = queue_out_of_sync_factor
        self.max_queue_size = int(self.batch_size * queue_out_of_sync_factor)

        # Get the right context for the multiprocessing:
        #     Fork seems to be the best, but is only avaliable on unix systems and does not support actors and learner
        #       on gpu
        #     Forkserver is then the second choice, Spawn the third
        fork_available = 'fork' in multiprocessing.get_all_start_methods() and self.policy.device == "cpu"
        forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
        if fork_available and start_method == 'fork':
            self.start_method_used = 'fork'
        elif forkserver_available and start_method == 'forkserver':
            self.start_method_used = 'forkserver'
        elif start_method == 'spawn':
            self.start_method_used = 'spawn'
        else:
            raise Exception('Please provide a valid start method. Options are for this system: {}'.format(
                multiprocessing.get_all_start_methods()
            ))
        ctx = multiprocessing.get_context(self.start_method_used)

        self.actor_output_queue = ctx.Queue(maxsize=self.max_queue_size)

        MyManager.register('BroadcastingContainer', BroadcastingContainer)
        manager = MyManager()
        manager.start()
        self.broadcasting_container = manager.BroadcastingContainer()

        self.actors = []
        for ii in range(self.n_actors):
            pickled_env_factory = cloudpickle.dumps(env_factory)
            pickled_policy = cloudpickle.dumps(self.policy)
            args = (pickled_env_factory, pickled_policy, n_rollout_steps,
                    self.actor_output_queue, self.broadcasting_container)
            self.actors.append(ctx.Process(target=_actor_worker, args=args))

    @override(BaseDistributedActors)
    def start(self):
        """Start all processes in the self.actors list"""
        for pp in self.actors:
            pp.start()

    @override(BaseDistributedActors)
    def stop(self):
        """Print kill all processes"""
        for pp in self.actors:
            if pp is not None and pp.is_alive():
                pp.terminate()
                pp.join()
        self.actor_output_queue.close()

    @override(BaseDistributedActors)
    def broadcast_updated_policy(self, state_dict: Dict) -> None:
        """Store the newest policy in the shared network object"""
        converted_state_dict = convert_to_torch(state_dict, in_place=False, cast=None, device=self.policy.device)
        self.broadcasting_container.set_policy_state_dict(converted_state_dict)

    @override(BaseDistributedActors)
    def collect_outputs(self, learner_device: str) -> Tuple[ActorOutput, float, float, float]:
        """Collect actor outputs from the multiprocessing queue."""
        actor_outputs = []
        episode_stats = []

        start_wait_time = time.time()
        q_size_before = self.actor_output_queue.qsize()

        while len(actor_outputs) < self.batch_size:
            try:
                tmp = self.actor_output_queue.get()
            except Empty:
                continue
            if isinstance(tmp[0], Exception):
                print('Original stack trace in process: ', tmp[1])
                raise tmp[0]
            actor_outputs.append(ActorOutput(*tmp[:-1]))
            episode_stats.append(tmp[-1])

        q_size_after = self.actor_output_queue.qsize()
        dequeue_time = time.time() - start_wait_time
        for episode_stats_actor in episode_stats:
            for stat in episode_stats_actor:
                if stat is not None:
                    self.epoch_stats.receive(stat)

        return batch_outputs_time_major(actor_outputs, learner_device=learner_device), q_size_before, q_size_after, \
               dequeue_time

    def __del__(self):
        print('Caught deletion, stopping actors')
        self.stop()


def _actor_worker(pickled_env_factory: bytes, pickled_policy: bytes,
                  n_rollout_steps: int, actor_output_queue: multiprocessing.Queue,
                  broadcasting_container: BroadcastingContainer):
    """Worker function for the actors. This Method is called with a new process. Its task is to initialize the,
        before going into a loop - updating its policy if necessary, computing a rollout and putting the result into
        the shared queue.

    :param pickled_env_factory: The pickled env factory
    :param pickled_policy: Pickled structured policy
    :param n_rollout_steps: the number of rollout steps to be computed for each rollout
    :param actor_output_queue: the queue to put the computed rollouts in
    :param broadcasting_container: the shared container, where actors can retrieve the newest version of the policies
    """
    try:
        actor = ActorAgent(
            env_factory=cloudpickle.loads(pickled_env_factory),
            policy=cloudpickle.loads(pickled_policy),
            n_rollout_steps=n_rollout_steps
        )
        policy_version_counter = 0

        while not broadcasting_container.stop_flag():
            # Update the policy if new version is available
            shared_policy_version_counter = broadcasting_container.policy_version()
            if policy_version_counter < shared_policy_version_counter:
                actor.update_policy(broadcasting_container.policy_state_dict())
                policy_version_counter = shared_policy_version_counter

            actor_output = actor.rollout()
            actor_output_queue.put([*actor_output])

    except Exception as e:
        actor_output_queue.put([e, traceback.format_exc()])
