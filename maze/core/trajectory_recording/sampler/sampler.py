"""File holdings the custom sampler for sampling the indices from a dataset."""
from typing import Iterable, Sized, Iterator

import torch
from torch.utils.data import Sampler, BatchSampler, IterDataPipe
from torch.utils.data.datapipes.datapipe import _IterDataPipeSerializationWrapper, MapDataPipe, \
    _MapDataPipeSerializationWrapper

from maze.core.annotations import override
from maze.core.env.structured_env import ActorID
from numpy.random._generator import Generator


class ActorIdSampler:
    """Sampler that randomly samples such as the entries of the bath obtained have all a consistent actor id
    :param data_source: Dataset to be used for sampling.
    :param generator: Generator used in sampling.
    """

    def __init__(self, data_source: Sized, generator: Generator | None = None):
        self.generator = generator
        self.data_source = data_source
        if isinstance(data_source, IterDataPipe):
            self.data_source = _IterDataPipeSerializationWrapper(data_source)
        elif isinstance(data_source, MapDataPipe):
            self.data_source = _MapDataPipeSerializationWrapper(data_source)

        self.indices = self._generate_indices()

    def _generate_indices(self) -> dict[int, list[int]]:
        """Generate the indices such as these will produce a consistent minibatch w.r.t. the actor id.
        :return: Mapping of the agent_id with the indices of their entries in the dataset.
        """
        indices = {}
        for idx, row in enumerate(self.data_source):
            assert len(row[-1]) == 1
            actor_id = row[-1][0]
            assert isinstance(actor_id, ActorID)
            if actor_id.agent_id not in indices:
                indices[actor_id.agent_id] = []
            indices[actor_id.agent_id].append(idx)
        return indices

    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    def get_iterators(self) -> list[Iterator[int]]:
        """Returns n iterators based on the keys of the indices dictionary."""
        iterators = []
        for value in self.indices.values():
            if self.generator is not None:
                self.generator.shuffle(value)
            iterators.append(iter(value))
        return iterators

    def __len__(self) -> int:
        """Overrides len method and returns the cumulative number of indices in the dataset.
        :return: Length of dataset
        """

        return sum(len(v) for v in self.indices.values())


class BatchActorIdSampler(BatchSampler):
    """Wraps a ActorIdSampler to yield a mini-batch of indices. Overrides torch.utils.data.sampler.BatchSampler
        Args:
            sampler (Sampler or Iterable): Base sampler. Can be any iterable object
            batch_size (int): Size of mini-batch.
            drop_last (bool): If ``True``, the sampler will drop the last batch if
                its size would be less than ``batch_size``
        """

    def __init__(self, sampler, batch_size: int, drop_last: bool):
        assert isinstance(sampler, ActorIdSampler)
        assert drop_last, f"drop_last set to {drop_last} is not yet supported."
        super().__init__(sampler, batch_size, drop_last)
        self.sampler: ActorIdSampler = sampler

    @override(BatchSampler)
    def __iter__(self) -> Iterator[list[int]]:
        """Overrides BatchSampler.__iter__ to yield a mini-batch of samples."""
        if self.drop_last:
            for sampler_iter in self.sampler.get_iterators():
                while True:
                    try:
                        batch = [next(sampler_iter) for _ in range(self.batch_size)]
                        yield batch
                    except StopIteration:
                        break
        else:
            raise NotImplementedError

