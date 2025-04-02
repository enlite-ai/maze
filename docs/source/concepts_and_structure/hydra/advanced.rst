.. |hydra_instantiation_functionality| raw:: html

   <a href="https://hydra.cc/docs/advanced/instantiate_objects/overview" target="_blank">Hydra's own instantiation functionality</a>

.. |github_omegaconf| raw:: html

   <a href="https://github.com/omry/omegaconf" target="_blank">OmegaConf</a>

.. |hydra_specializing_config| raw:: html

   <a href="https://hydra.cc/docs/patterns/specializing_config" target="_blank">in Hydra docs here</a>

.. _hydra-advanced:

Hydra: Advanced Concepts
========================

This page features a collection of more advanced Maze and Hydra features
that are used across the framework and power the configuration under the hood:

- :ref:`Factory<hydra-advanced-factory>`, which Maze uses to turn configuration
  into instantiated objects, while allowing passing in already instantiated objects as well.
- :ref:`Interpolation<hydra-advanced-interpolation>`, which allows you to reference
  parts of configuration from elsewhere.
- :ref:`Specializations<hydra-advanced-specializations>`, which allow you
  to load additional configuration files based on particular combinations
  of selected defaults.


.. _hydra-advanced-factory:

Maze Factory
------------

:class:`Factory<maze.core.utils.factory.Factory>` wraps around
|hydra_instantiation_functionality| and adds features like
type hinting and checking, collections, configuration structure checks,
and ability to take in already instantiated objects.

Using the factory, classes can accept
:class:`ConfigType<maze.core.utils.factory.ConfigType>` (or collections thereof,
:class:`CollectionOfConfigType<maze.core.utils.factory.CollectionOfConfigType>`),
which stands for either an already instantiated object, or a dictionary
with configuration, which the factory will then use to build the instance.

Configuration dictionary consists of the ``_target_`` attribute, along with any
arguments that the instantiated class takes, e.g. (here denoted in YAML, as you
will find it in many places across the framework):

.. literalinclude:: code_snippets/advanced_factory_dict.yaml
  :language: yaml

The factory then takes in the dictionary configuration (loaded from YAML
using Hydra, or from anywhere else) and builds the object for you,
checking that it is indeed of the expected type:

.. literalinclude:: code_snippets/advanced_factory_example.py
  :language: python

You can also pass in additional keyword arguments that the factory will
then pass on to the constructor together with anything from the configuration dictionary:

.. literalinclude:: code_snippets/advanced_factory_example_kwargs.py
  :language: python

If you pass in an already instantiated object instead of a configuation dictionary,
the ``instantiate`` method will only check that it is of the expected type
and return it back. This allows components in Maze to be easily configurable
both from YAML/dictionaries and by passing in already instantiated objects.


.. _hydra-advanced-interpolation:

Interpolation
-------------

Hydra is based on |github_omegaconf| and supports interpolation.

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

Specializations are well explained |hydra_specializing_config|.


Where to Go Next
----------------

After understanding advanced Hydra configuration, you might want to:

- :ref:`Create custom Hydra configuration files<hydra-custom>` for your project
- Review the root configurations available in the Maze framework (as they are a good basis
  for your custom configurations)

