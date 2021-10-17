"""Actors distributed across multiple processing using multiprocessing."""

import time
from multiprocessing.managers import BaseManager
from typing import Callable, Union, Tuple, Dict, List

import cloudpickle
from torch import multiprocessing

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.core.rollout.rollout_generator import RolloutGenerator
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.core.trajectory_recording.records.trajectory_record import SpacesTrajectoryRecord
from maze.perception.perception_utils import convert_to_torch
from maze.train.parallelization.distributed_actors.broadcasting_container import BroadcastingContainer
from maze.train.parallelization.distributed_actors.distributed_actors import DistributedActors
from maze.utils.exception_report import ExceptionReport


class MyManager(BaseManager):
    """Basic wrapper for the BaseManager class"""
    pass


class SubprocDistributedActors(DistributedActors):
    """Basic Distributed-Actors-Module using python multiprocessing.Process

    :param queue_out_of_sync_factor: this factor multiplied by the actor_batch_size gives the size of the queue.
        Therefore if the all rollouts computed can be  at most
        (queue_out_of_sync_factor + num_agents/actor_batch_size) out of sync with learner policy.
    :param start_method: Method used to start the subprocesses.
       Must be one of the methods returned by multiprocessing.get_all_start_methods().
       Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    :param actor_env_seeds: A list of seeds for each actors' env.
    :param actor_agent_seeds: A list of seed for each actors' policy.
    """

    def __init__(self,
                 env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv]],
                 policy: TorchPolicy,
                 n_rollout_steps: int,
                 n_actors: int,
                 batch_size: int,
                 queue_out_of_sync_factor: float,
                 start_method: str,
                 actor_env_seeds: List[int],
                 actor_agent_seeds: List[int]):
        super().__init__(env_factory, policy, n_rollout_steps, n_actors, batch_size)

        self.queue_out_of_sync_factor = queue_out_of_sync_factor
        self.max_queue_size = int(self.batch_size * queue_out_of_sync_factor)

        ctx = self.get_multiprocessing_context(start_method)
        self.actor_output_queue = ctx.Queue(maxsize=self.max_queue_size)

        MyManager.register('BroadcastingContainer', BroadcastingContainer)
        manager = MyManager()
        manager.start()
        self.broadcasting_container = manager.BroadcastingContainer()

        self.actors = []
        for env_seed, agent_seed in zip(actor_env_seeds, actor_agent_seeds):
            pickled_env_factory = cloudpickle.dumps(env_factory)
            pickled_policy = cloudpickle.dumps(self.policy)
            args = (pickled_env_factory, pickled_policy, n_rollout_steps,
                    self.actor_output_queue, self.broadcasting_container, env_seed, agent_seed)
            self.actors.append(ctx.Process(target=_actor_worker, args=args))

    @override(DistributedActors)
    def start(self):
        """Start all processes in the self.actors list"""
        for pp in self.actors:
            pp.start()

    @override(DistributedActors)
    def stop(self):
        """Print kill all processes"""
        for pp in self.actors:
            if pp is not None and pp.is_alive():
                pp.terminate()
                pp.join()
        self.actor_output_queue.close()

    @override(DistributedActors)
    def broadcast_updated_policy(self, state_dict: Dict) -> None:
        """Store the newest policy in the shared network object"""
        converted_state_dict = convert_to_torch(state_dict, in_place=False, cast=None, device=self.policy.device)
        self.broadcasting_container.set_policy_state_dict(converted_state_dict)

    @override(DistributedActors)
    def collect_outputs(self, learner_device: str) -> Tuple[StructuredSpacesRecord, float, float, float]:
        """Collect actor outputs from the multiprocessing queue."""
        trajectories = []

        start_wait_time = time.time()
        q_size_before = self.actor_output_queue.qsize()

        while len(trajectories) < self.batch_size:
            trajectory_report: Union[SpacesTrajectoryRecord, ExceptionReport] = self.actor_output_queue.get()
            if isinstance(trajectory_report, ExceptionReport):
                raise RuntimeError("An actor encountered the following error:\n"
                                   + trajectory_report.traceback) from trajectory_report.exception

            trajectories.append(trajectory_report)

            # collect episode statistics
            for record in trajectory_report.step_records:
                if record.episode_stats is not None:
                    self.epoch_stats.receive(record.episode_stats)

        q_size_after = self.actor_output_queue.qsize()
        stacked_record = SpacesTrajectoryRecord.stack_trajectories(trajectories).stack()
        dequeue_time = time.time() - start_wait_time

        return stacked_record.to_torch(device=learner_device), q_size_before, q_size_after, dequeue_time

    def __del__(self):
        print('Caught deletion, stopping actors')
        self.stop()

    def get_multiprocessing_context(self, start_method: str):
        """Get the right context for the multiprocessing.

        Fork is the best option, but is only available on unix systems and does not support actors and learner
        on gpu. Forkserver is then the second choice, Spawn the third.
        :return:
        """

        fork_available = 'fork' in multiprocessing.get_all_start_methods() and self.policy.device == "cpu"
        forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
        if fork_available and start_method == 'fork':
            start_method_used = 'fork'
        elif forkserver_available and start_method == 'forkserver':
            start_method_used = 'forkserver'
        elif start_method == 'spawn':
            start_method_used = 'spawn'
        else:
            raise Exception('Please provide a valid start method. Options are for this system: {}'.format(
                multiprocessing.get_all_start_methods()
            ))

        return multiprocessing.get_context(start_method_used)


def _actor_worker(pickled_env_factory: bytes, pickled_policy: bytes,
                  n_rollout_steps: int, actor_output_queue: multiprocessing.Queue,
                  broadcasting_container: BroadcastingContainer, env_seed: int, agent_seed: int):
    """Worker function for the actors. This Method is called with a new process. Its task is to initialize the,
        before going into a loop - updating its policy if necessary, computing a rollout and putting the result into
        the shared queue.

    :param pickled_env_factory: The pickled env factory
    :param pickled_policy: Pickled structured policy
    :param n_rollout_steps: the number of rollout steps to be computed for each rollout
    :param actor_output_queue: the queue to put the computed rollouts in
    :param broadcasting_container: the shared container, where actors can retrieve the newest version of the policies
    :param env_seed: The env seed to be used.
    :param agent_seed: The agent seed to be used.
    """
    try:
        env_factory = cloudpickle.loads(pickled_env_factory)
        env = env_factory()
        env.seed(env_seed)

        policy: TorchPolicy = cloudpickle.loads(pickled_policy)
        policy.seed(agent_seed)
        policy_version_counter = -1

        rollout_generator = RolloutGenerator(env=env, record_logits=True, record_episode_stats=True)

        while not broadcasting_container.stop_flag():
            # Update the policy if new version is available
            shared_policy_version_counter = broadcasting_container.policy_version()
            if policy_version_counter < shared_policy_version_counter:
                policy.load_state_dict(broadcasting_container.policy_state_dict())
                policy_version_counter = shared_policy_version_counter

            trajectory = rollout_generator.rollout(policy, n_steps=n_rollout_steps)
            actor_output_queue.put(trajectory)

    except Exception as e:
        actor_output_queue.put(ExceptionReport(e))
