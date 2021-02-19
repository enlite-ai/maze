"""Runner implementations for Evolution Strategies"""
from abc import abstractmethod
from dataclasses import dataclass
from typing import Union

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.common.training_runner import TrainingRunner
from maze.train.trainers.es.distributed.es_distributed_rollouts import ESDistributedRollouts
from maze.train.trainers.es.distributed.es_dummy_distributed_rollouts import ESDummyDistributedRollouts
from maze.train.trainers.es.es_shared_noise_table import SharedNoiseTable
from maze.train.trainers.es.es_trainer import ESTrainer
from omegaconf import DictConfig


@dataclass
class ESMasterRunner(TrainingRunner):
    """Baseclass of ES training master runners (serves as basis for dev and other runners)."""

    shared_noise_table_size: int
    """Number of float values in the deterministically generated pseudo-random table
    (250.000.000 x 32bit floats = 1GB)"""

    @override(TrainingRunner)
    def run(self, cfg: DictConfig) -> None:
        """Run the training master node."""
        super().run(cfg)
        env = self.env_factory()

        # --- init the shared noise table ---
        print("********** Init Shared Noise Table **********")
        shared_noise = SharedNoiseTable(count=self.shared_noise_table_size)

        # --- initialize policies ---
        policy = TorchPolicy(networks=self.model_composer.policy.networks,
                             distribution_mapper=self.model_composer.distribution_mapper,
                             device="cpu")

        print("********** Trainer Setup **********")
        trainer = ESTrainer(algorithm_config=cfg.algorithm,
                            policy=policy,
                            shared_noise=shared_noise,
                            normalization_stats=self.normalization_statistics)

        # initialize model from input_dir
        self._init_trainer_from_input_dir(trainer=trainer, state_dict_dump_file=self.state_dict_dump_file,
                                          input_dir=cfg.input_dir)

        model_selection = BestModelSelection(dump_file=self.state_dict_dump_file, model=policy)

        print("********** Run Trainer **********")
        # run with pseudo-distribution, without worker processes
        trainer.train(self.create_distributed_rollouts(env=env, shared_noise=shared_noise),
                      model_selection=model_selection)

    @abstractmethod
    def create_distributed_rollouts(self,
                                    env: Union[StructuredEnv, StructuredEnvSpacesMixin],
                                    shared_noise: SharedNoiseTable) -> ESDistributedRollouts:
        """Abstract method, derived runners like ESDevRunner return an appropriate rollout generator.

        :param env: the one and only environment
        :param shared_noise: noise table to be shared by all workers
        :return: a newly instantiated rollout generator
        """


@dataclass
class ESDevRunner(ESMasterRunner):
    """Runner config for single-threaded training, based on ESDummyDistributedRollouts."""

    n_eval_rollouts: int
    """Fixed number of evaluation runs per epoch."""

    def create_distributed_rollouts(self,
                                    env: Union[StructuredEnv, StructuredEnvSpacesMixin],
                                    shared_noise: SharedNoiseTable) -> ESDistributedRollouts:
        """use single-threaded rollout generation"""
        return ESDummyDistributedRollouts(env=env, shared_noise=shared_noise, n_eval_rollouts=self.n_eval_rollouts)


