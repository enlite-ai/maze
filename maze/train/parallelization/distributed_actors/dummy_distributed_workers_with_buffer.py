"""Dummy distributed workers. Ran sequentially in the main process."""
import time
from typing import Tuple, Dict

from maze.core.annotations import override
from maze.core.rollout.rollout_generator import RolloutGenerator
from maze.perception.perception_utils import convert_to_torch
from maze.train.parallelization.broadcasting_container import BroadcastingContainer
from maze.train.parallelization.distributed_actors.base_distributed_workers_with_buffer import \
    BaseDistributedWorkersWithBuffer


class DummyDistributedWorkersWithBuffer(BaseDistributedWorkersWithBuffer):
    """Dummy implementation of distributed workers with buffer creates the workers as a list. Once the outputs are to
        be collected, it simply rolls them out in a loop until is has enough to be added to the buffer.
    """

    @override(BaseDistributedWorkersWithBuffer)
    def _init_workers(self):

        self.broadcasting_container = BroadcastingContainer()
        self.current_worker_idx = 0

        self.workers = []
        self.policy_version_counter = 0

        for env_seed in self.env_instance_seeds:
            env = self.env_factory()
            env.seed(env_seed)
            actor = RolloutGenerator(env=env, record_next_observations=True, record_episode_stats=True)
            self.workers.append(actor)

    @override(BaseDistributedWorkersWithBuffer)
    def start(self) -> None:
        """Nothing to do in dummy implementation"""
        pass

    @override(BaseDistributedWorkersWithBuffer)
    def stop(self) -> None:
        """Nothing to do in dummy implementation"""
        pass

    @override(BaseDistributedWorkersWithBuffer)
    def broadcast_updated_policy(self, state_dict: Dict) -> None:
        """Store the newest policy in the shared network object"""
        converted_state_dict = convert_to_torch(state_dict, in_place=False, cast=None,
                                                device=self._worker_policy.device)
        self.broadcasting_container.set_policy_state_dict(converted_state_dict)

    @override(BaseDistributedWorkersWithBuffer)
    def collect_rollouts(self) -> Tuple[float, float, float]:
        """implementation of
        :class:`~maze.train.parallelization.distributed_actors.base_distributed_workers_with_buffer.BaseDistributedWorkersWithBuffer`
        interface
        """

        assert len(self.replay_buffer) >= self.batch_size, \
            f'The replay buffer should hold more transitions ({len(self.replay_buffer)}) than the batch size ' \
            f'({self.batch_size}) at all times'

        start_wait_time = time.time()

        for i in range(self.rollouts_per_iteration):
            # Update the policy if a new version of the policy has been published by the learner
            shared_policy_version_counter = self.broadcasting_container.policy_version()
            if self.policy_version_counter < shared_policy_version_counter:
                self._worker_policy.load_state_dict(self.broadcasting_container.policy_state_dict())
                self.policy_version_counter = shared_policy_version_counter

            trajectory = self.workers[self.current_worker_idx].rollout(policy=self._worker_policy,
                                                                       n_steps=self.n_rollout_steps)

            if self.split_rollouts_into_transitions:
                self.replay_buffer.add_rollout(trajectory)
            else:
                self.replay_buffer.add_transition(trajectory)

            self.current_worker_idx = self.current_worker_idx + 1 if self.current_worker_idx < len(
                self.workers) - 1 else 0

            # collect episode statistics
            for step_record in trajectory.step_records:
                if step_record.episode_stats is not None:
                    self.epoch_stats.receive(step_record.episode_stats)

        dequeue_time = time.time() - start_wait_time

        return 0, 0, dequeue_time
