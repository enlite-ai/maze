"""ES shared noise implementation based on https://github.com/openai/evolution-strategies-starter"""

from __future__ import annotations

import ctypes
import itertools
import multiprocessing

import numpy as np


class SharedNoiseTable:
    """A fixed length vector of deterministically generated pseudo-random floats.

    This enables a communication strategy for the distributed training, that allows to transfer noise table indices
    instead of full gradient vectors.

    :param count: Number of float values in the fixed length table (250.000.000 x 32bit floats = 1GB)
    """

    def __init__(self, count: int = 250_000_000, context=None):
        seed = 123
        # default is 1 gigabyte of 32-bit numbers
        print(f'Sampling {count} random numbers with seed {seed}')

        # We use lock=False to avoid the SemLock error in Python 3.12
        if context is None:
            self._shared_mem = multiprocessing.Array(ctypes.c_float, count, lock=False)
        else:
            self._shared_mem = context.Array(ctypes.c_float, count, lock=False)

        # Check if the object has 'get_obj' (it won't if lock=False)
        if hasattr(self._shared_mem, 'get_obj'):
            self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())
        else:
            # It's already the raw array
            self.noise = np.ctypeslib.as_array(self._shared_mem)

        assert self.noise.dtype == np.float32

        # split into smaller chunks, allocation of all at once would need 3 times the memory of the table
        # (noise table + numpy array as 64 bit floats)
        batch_size = 10_000_000
        for i in itertools.count():
            start = i * batch_size
            end = min(count, start + batch_size)

            # 64-bit to 32-bit conversion here
            self.noise[start:end] = np.random.RandomState(seed).standard_normal(end - start)

            if end == count:
                break

        print(f'Sampled {self.noise.size * 4} bytes')

    def get(self, i: int, dim: int) -> np.ndarray:
        """Get the pseudo-random sequence at table index i.

        :param i: start index within the table
        :param dim: desired vector length

        :return A noise vector with length dim
        """

        # if the index is near the end of the table we might need to concatenate the vector
        noise = []

        end_idx = i + dim
        slice_start_idx = i
        while True:
            slice_end_idx = min(end_idx, len(self.noise))
            noise.append(self.noise[slice_start_idx:slice_end_idx])
            if end_idx <= len(self.noise):
                break

            # we need to cycle once more through the table
            slice_start_idx = 0
            end_idx -= len(self.noise)

        # stack the parts together
        return np.concatenate(noise)

    def sample_index(self, rng: np.random.RandomState) -> int:
        """Sample a random index within the table, taking into account the size of the noise vector.

        :param rng: Maze random number generator to be used.

        :return: A noise index to be passed to
                 :meth:`maze.train.trainers.es.es_shared_noise_table.SharedNoiseTable.get`.
        """
        return rng.randint(0, len(self.noise))
