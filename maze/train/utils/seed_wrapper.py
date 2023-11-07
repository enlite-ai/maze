"""File holding a Seed wrapper for training."""
from typing import Any


class SeedWrapper:
    """A wrapper for the seeds, such that the same seed can be used multiple times."""

    def __init__(self, seed: Any, random_seed: int):
        self.seed = seed
        self.random_seed = random_seed

    def get_seed(self) -> Any:
        """Return the seed."""
        return self.seed

    def __repr__(self) -> str:
        """Get a string representation of the instance."""
        return f'{self.seed} + {self.random_seed}'
