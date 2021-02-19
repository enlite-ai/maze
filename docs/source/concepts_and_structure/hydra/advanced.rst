.. _hydra-advanced:

Hydra: Advanced Concepts
========================

This page features a collection of more advanced Hydra features
which are used throughout the framework.

.. _hydra-advanced-interpolation:

Interpolation
-------------

Hydra is based on `OmegaConf`_ and supports interpolation.

.. _`OmegaConf`: https://github.com/omry/omegaconf

Interpolation allows us to reference and reuse a value defined elsewhere in the
configuration, without repeating it. For example:

.. code-block:: yaml

  original:
    value: 1  # We want to reference this value elsewhere
  some:
    other:
      structure: ${original.value}  # Reference

A (somewhat limited) form of interpolation is used also in specializations described below.


.. _hydra-advanced-specializations:

Specializations
---------------

Specializations are parts of config that depend on multiple components.
For example, your wrapper configuration might depend on both
the environment chosen (e.g., ``gym_pixel_env`` or ``gym_feature_env``) and
your model (e.g., ``default`` or ``rnn``) -- if using an RNN, you might want
to include `ObservationStackWrapper`, but its configuration also depends on the environment used.

Then, specializations come to the rescue. In your root config file, you can include
a specialization like this (for illustrative purposes):

.. code-block:: yaml

  defaults:
    - env: gym_pixel_env
    - model: default
    - env_model: ${defaults.1.env}-${defaults.2.model}
      optional: true

Then, when you run this configuration with ``env=gym_pixel_env`` and ``model=rnn``,
Hydra will look into the ``env_model`` directory for configuration named ``gym_pixel_env-rnn.yaml``.
This allows you to capture the dependencies between these two components easily without
having to specify more overrides.

Specializations are well explained  `in Hydra docs here`_.

.. _`in Hydra docs here`: https://hydra.cc/docs/next/patterns/specializing_config


Where to Go Next
----------------

After understanding advanced Hydra configuration, you might want to:

- :ref:`Create custom Hydra configuration files<hydra-custom>` for your project
- Review the root configurations available in the Maze framework (as they are a good basis
  for your custom configurations)

