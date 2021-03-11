.. _experimenting:

Experiment Configuration
========================

Launching experiments with the Maze command line interface (CLI)
is based on the :ref:`Hydra configuration system <hydra>` and hence also closely follows Hydra's experimentation
workflow. In general, there are different options for carrying out and configuring experiments with Maze.
(To see experiment configuration in action, check out our
`project template <https://github.com/enlite-ai/maze-cartpole>`_.)

.. contents:: Overview
    :depth: 1
    :local:
    :backlinks: top

Command Line Overrides
----------------------

To quickly play around with parameters in an interactive (temporary) fashion
you can utilize Hydra command line overrides to reset parameters specified
in the default config
(e.g., `conf_train <https://github.com/enlite-ai/maze/blob/main/maze/conf/conf_train.yaml/>`_).

.. code:: console

   $ maze-run -cn conf_train env.name=CartPole-v0 algorithm=ppo algorithm.lr=0.0001

The example above changes the trainer to PPO and optimizes with a learning rate of 0.0001.
You can of course override any other parameter of your training and rollout runs.

For an in depth explanation of the override concept we refer to our
:ref:`Hydra documentation <hydra-overview-overrides>`.

.. _experimenting_files:

Experiment Config Files
-----------------------

For a more persistent way of structuring your experiments you can also make use of
`Hydra's built-in Experiment Configuration <https://hydra.cc/docs/next/patterns/configuring_experiments/>`_.

This allows you to maintain multiple experimental config files
each only specifying the changes to the default config
(e.g., `conf_train <https://github.com/enlite-ai/maze/blob/main/maze/conf/conf_train.yaml/>`_).

.. literalinclude:: ../../../maze/conf/experiment/cartpole_ppo_wrappers.yaml
  :language: YAML
  :caption: conf/experiment/cartpole_ppo_wrappers.yaml

The experiment config above sets the trainer to PPO, the learning rate to 0.0001
and additionally activates the
`vector_obs <https://github.com/enlite-ai/maze/blob/main/maze/conf/wrappers/vector_obs.yaml/>`_ wrapper stack.

To start the training run with this config file, run:

.. code:: console

   $ maze-run -cn conf_train +experiment=cartpole_ppo_wrappers

You can find a more detail explanation on how experiments are embedded in the overall configuration system in our
:ref:`Hydra experiment documentation <hydra-custom-experiments>`.

Where to Go Next
----------------

- Here you can learn how to set up a :ref:`custom configuration/experimentation module <hydra-custom-search_path>`.
- If you would like to learn about more advanced configuration options you can dive into the
  :ref:`Hydra configuration system documentation<hydra>`.
- Read up on :ref:`training <training>` and :ref:`rollouts <rollouts>`.
- Clone this `project template repo <https://github.com/enlite-ai/maze-cartpole>`_ to start your own Maze project.