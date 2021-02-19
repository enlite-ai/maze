.. _distributions_reference:

Action Spaces and Distributions Module
======================================

This page contains the reference documentation of
:ref:`Maze Action Spaces and Distributions Module <action_spaces_and_distributions_module>`.


These are interfaces, classes and utility functions:

.. currentmodule:: maze.distributions

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~distribution.ProbabilityDistribution
    ~torch_dist.TorchProbabilityDistribution
    ~distribution_mapper.DistributionMapper
    ~utils.atanh
    ~utils.tensor_clamp

These are built-in Torch probability distributions:

.. currentmodule:: maze.distributions

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~categorical.CategoricalProbabilityDistribution
    ~bernoulli.BernoulliProbabilityDistribution
    ~gaussian.DiagonalGaussianProbabilityDistribution
    ~squashed_gaussian.SquashedGaussianProbabilityDistribution
    ~beta.BetaProbabilityDistribution

These are combined probability distributions:

.. currentmodule:: maze.distributions

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~multi_categorical.MultiCategoricalProbabilityDistribution
    ~dict.DictProbabilityDistribution
