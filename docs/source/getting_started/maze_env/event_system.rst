.. |tutorial_code_part_04| raw:: html

   <a href="https://github.com/enlite-ai/maze-examples/tree/main/tutorial_maze_env/part04_events" target="_blank">can be found here</a>

.. _env_from_scratch-events:

Adding Events and KPIs
======================

The complete code for this part of the tutorial |tutorial_code_part_04|

.. code:: bash

    # file structure
    - cutting_2d
        - main.py  # modified
        - env
            - core_env.py  # modified
            - inventory.py  # modified
            - maze_state.py
            - maze_action.py
            - renderer.py
            - maze_env.py
            - events.py  # new
            - kpi_calculator.py  # new
        - space_interfaces
            - dict_action_conversion.py
            - dict_observation_conversion.py
        - conf ...

.. _env_from_scratch-events-events:

Events
------

In the :ref:`previous section <train_maze_env>` we have trained the initial version of our cutting environment
and already learned how we can watch the training process with
:ref:`commandline and Tensorboard logging <logging>`.
However, watching only standard metrics such as *reward* or *episode step count* is not always too informative
with respect to the agents behaviour and the problem at hand.

For example we might be interested in how often an agent selects an invalid cutting piece or
specifies and invalid cutting setting.
To tackle this issue and to enable better inspection and logging tools we introduce
an :ref:`event system <event_system>` that will be also reused in
the :ref:`reward customization section <tutorial-reward>` of this tutorial.

In particular, we introduce two event types related to the cutting process as well as inventory management.
For each event we can define which statistics are computed at which stage of the aggregation process
(*event*, *step*, *epoch*) via event decorators:

- :code:`@define_step_stats(len)`: Events :math:`e_i` are collected as a list of events :math:`\{e_i\}`.
  The ``len`` function counts how often such an event occurred in the current environment step
  :math:`Stats_{Step}=|\{e_i\}|`.
- :code:`@define_episode_stats(sum)`: Defines how the :math:`S` step statistics
  should be aggregated to episode statistics by simply summing them up: :math:`Stats_{Episode}=\sum^S Stats_{Step}`
- :code:`@define_epoch_stats(np.mean, output_name="mean_episode_total")`: A training epoch consists of N episodes.
  This decorator defines that epoch statistics should be the average of the contained episodes:
  :math:`Stats_{Epoch}=(\sum^N Stats_{Episode})/N`

:ref:`Below <env_from_scratch-events-main>` we will see that theses statistics will now be considered by the logging
system as *InventoryEvents* and *CuttingEvents*.
For more details on event decorators and the underlying working principles we refer to
the dedicated section on :ref:`event and KPI logging <event_kpi_log>`.

.. literalinclude:: ../../../../tutorials/tutorial_maze_env/part04_events/env/events.py
  :language: python
  :caption: env/events.py

.. _env_from_scratch-events-kpi:

KPI Calculator
--------------

The goal of the cutting 2d environment is to learn a cutting policy that requires as little as possible
raw inventory pieces for fulfilling upcoming customer demand.
This metric is exactly what we :ref:`define as the KPI <event_kpi_log-kpis>` to watch and optimize,
e.g. the **raw_piece_usage_per_step**.

As you will see :ref:`below <env_from_scratch-events-main>` the logging system considers such KPIs
and prints statistics of these along with the remaining *BaseEnvEvents*.

.. literalinclude:: ../../../../tutorials/tutorial_maze_env/part04_events/env/kpi_calculator.py
  :language: python
  :caption: env/kpi_calculator.py

Updating CoreEnv and Inventory
------------------------------

There are also a few changes we have to make in the CoreEnvironment:

- initialize the Publisher-Subscriber and the KPI Calculator
- creating the event topics for cutting and inventory events when setting up the environment
- instead of writing relevant events into the info dictionary in the step function
  we can now trigger the respective events.

.. literalinclude:: ../../../../tutorials/tutorial_maze_env/part04_events/env/core_env_snippet.py
  :language: python
  :caption: env/core_env.py

For the inventory we proceed analogously and also trigger the respective events.

.. literalinclude:: ../../../../tutorials/tutorial_maze_env/part04_events/env/inventory_snippet.py
  :language: python
  :caption: env/inventory.py

.. _env_from_scratch-events-main:

Test Script
-----------

The following snippet will instantiate the environment and run it for 15 steps.

To get access to event and KPI logging we need to wrap the environment with the
:class:`~maze.core.wrappers.log_stats_wrapper.LogStatsWrapper`.
To simplify the statistics logging setup we rely on the
:class:`~maze.utils.log_stats_utils.SimpleStatsLoggingSetup` helper class.

.. literalinclude:: ../../../../tutorials/tutorial_maze_env/part04_events/main.py
  :language: python
  :caption: main.py

When running the script you will get an output as shown below.
Note that statistics of both, events and KPIs, are printed along with default *reward* or *action* statistics.

.. literalinclude:: cmd_event_log.log
  :language: bash