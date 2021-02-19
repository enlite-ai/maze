"""Contains the basic interface definitions for probability distributions."""
from abc import ABC
from typing import Any


class ProbabilityDistribution(ABC):
    """Base class for all probability distributions.
    """

    def neg_log_prob(self, actions: Any) -> Any:
        """Returns the the negative log likelihood of the provided actions.

        :param actions: the actions.
        :return: negative log likelihood tensor.
        """
        return - self.log_prob(actions)

    def log_prob(self, actions: Any) -> Any:
        """Returns the the log likelihood of the provided actions.

        actions: the actions.
        :return: log likelihood tensor.
        """
        raise NotImplementedError

    def entropy(self) -> Any:
        """Calculate the entropy of the probability distribution.

        :return: entropy tensor.
        """
        raise NotImplementedError

    def kl(self, other: 'ProbabilityDistribution') -> Any:
        """Calculates the Kullback-Leibler between self and the other probability distribution.

        :param other: ([float]) the distribution to compare with.
        :return: kl tensor.
        """
        raise NotImplementedError

    def sample(self) -> Any:
        """Draw a sample from the probability distribution.

        :return: stochastic sample tensor.
        """
        raise NotImplementedError

    def deterministic_sample(self) -> Any:
        """Draw a deterministic sample from the probability distribution.

        :return: deterministic sample tensor.
        """
        raise NotImplementedError
