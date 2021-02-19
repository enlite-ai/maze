.. _obs_logging:

Observation Distribution Visualization
======================================

Maze provides the option to watch the evolution of value ranges of observations throughout the training process.
This is especially useful for debugging your experiments and training runs as it reveals if:

- observations stay within an expected value range.
- observation normalization is applied correctly.
- observations drift as the agent's behaviour evolves throughout training.

Activating Observation Logging
------------------------------

To activate observation logging you only have to add the
:class:`~maze.core.wrappers.observation_logging_wrapper.ObservationLoggingWrapper`
to your environment wrapper stack in your yaml config:

.. code:: YAML

    # @package wrappers
    ObservationLoggingWrapper: {}

If you are using plain Python you can start with the code snippet below.

.. code:: PYTHON

    from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
    from maze.core.wrappers.observation_logging_wrapper import ObservationLoggingWrapper

    env = GymMazeEnv(env="CartPole-v0")
    env = ObservationLoggingWrapper(env)

For both cases observations will be logged and distribution plots will be added to Tensorboard.

.. warning::

    We support observation logging as an opt-in feature via a dedicated wrapper
    and recommend to use it only for debugging and inspection purposes.
    Once everything is on track and training works as expected we suggest to remove the wrapper again
    especially when dealing with environments with large observations.
    If you forget to remove it training might get slow and the memory consumption of Tensorboard might explode!

Tensorboard Examples
--------------------

Maze visualizes observations on a per-epoch basis in the *DISTRIBUTIONS* and *HISTOGRAMS* tab of Tensorboard.
By using the slider above the images you can step through the training epochs and see how the observation distribution
evolves over time.

Below you see an example for both versions (just click the figure to view it in large).

.. image:: img/tb_obs_distributions.png
   :width: 49 %

.. image:: img/tb_obs_histogram.png
   :width: 49 %

Note that two different versions of the observation distribution are logged:

- *observation_original:* distribution of the original observation returned by the environment.
- *observation_processed:* distribution of the observation after processing
  (e.g. :ref:`pre-processing <observation_pre_processing>` or :ref:`normalization <observation_normalization>`).

This is especially useful to verify if the applied observation processing steps yield the expected result.

Where to Go Next
----------------

- You might be also interested in :ref:`logging action distributions <act_logging>`.
- You can learn more about :ref:`observation pre-processing <observation_pre_processing>`
  and :ref:`observation normalization <observation_normalization>`.