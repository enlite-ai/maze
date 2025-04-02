.. |tutorial_code_part_05| raw:: html

   <a href="https://github.com/enlite-ai/maze-examples/tree/main/tutorial_maze_env/part05_reward" target="_blank">can be found here</a>

.. _tutorial-reward:

Adding Reward Customization
===========================

The complete code for this part of the tutorial |tutorial_code_part_05|

.. code:: bash

    # file structure
    - cutting_2d
        - main.py  # modified
        - env
            - core_env.py  # modified
            - inventory.py
            - maze_state.py
            - maze_action.py
            - renderer.py
            - maze_env.py  # modified
            - events.py
            - kpi_calculator.py
        - space_interfaces
            - dict_action_conversion.py
            - dict_observation_conversion.py
        - reward
            - default_reward.py  # new

Reward
------

In this part of the tutorial we introduce how to reuse the event system for
:ref:`reward shaping and customization <reward_aggregation>` via the
:class:`~maze.core.env.reward.RewardAggregatorInterface`.

In Maze, reward aggregators usually calculate reward from the current environment
state, events that happened during the last step, or a combination thereof. Calculating
reward from state is generally simpler, but not a good fit for this environment -- here,
the reward is more concerned with what happened (was an invalid cut attempted? A new raw piece replenished?)
than with the current state (i.e., the inventory state after the step). Hence,
the reward calculation here is based on events (which is in general more
flexible than using the environment state only).

The ``DefaultRewardAggregator`` does the following:

- Requests the required event interfaces via ``get_interfaces`` (here *CuttingEvents* and *InventoryEvents*).
- Collects rewards and penalties according to relevant events.
- Aggregates the individual event rewards and penalties to a single scalar reward signal.

Note that this reward aggregator can have any form as long as it provides a scalar reward function
that can be used for training. This gives a lot of flexibility in shaping rewards without the need to change the
actual implementation of the environment (:ref:`more on this topic <reward_aggregation>`).

.. literalinclude:: ../../../../tutorials/tutorial_maze_env/part05_reward/reward/default_reward.py
  :language: python
  :caption: reward/default_reward.py

Updating the Core- and MazeEnv
------------------------------

We also have to make a few modifications in the ``CoreEnv``:

- Initialize the reward aggregator in the constructor.
- Instead of accumulating reward in the if-else branches of the ``step`` function we summarize it only once
  at the end.

.. literalinclude:: ../../../../tutorials/tutorial_maze_env/part05_reward/env/core_env_snippet.py
  :language: python
  :caption: env/core_env.py

Finally, we update the ``maze_env_factory`` function for instantiating the trainable ``MazeEnv``
and we are all set up for training with event based, customized rewards.

.. literalinclude:: ../../../../tutorials/tutorial_maze_env/part05_reward/env/maze_env_snippet.py
  :language: python
  :caption: env/maze_env.py

Where to Go Next
----------------

As the reward is implemented via a reward aggregator that is methodologically identical to the initial version
there is no need to retrain the model for now.
However, we highly recommend to proceed with the more advanced tutorial on
:ref:`Structured Environments and Action Masking <struct_env_tutorial>`.