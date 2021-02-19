.. _env_from_scratch-maze_env:

Implementing the MazeEnv
========================

The complete code for this part of the tutorial
`can be found here <https://github.com/enlite-ai/maze/tree/main/tutorials/tutorial_maze_env/part03_maze_env>`_

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
            - maze_env.py  # new
        - space_interfaces
            - dict_action_conversion.py  # new
            - dict_observation_conversion.py  # new

.. contents:: Page Overview
    :depth: 1
    :local:
    :backlinks: top

MazeEnv
-------

The :ref:`MazeEnv <env_hierarchy-maze_env>` wraps the CoreEnvs as a Gym-style environment
in a reusable form, by utilizing the :ref:`interfaces (mappings) <env_hierarchy-interfaces>`
from the MazeState to the observation and from the MazeAction to the action.
After implementing the MazeEnv we will be ready to perform our first training run.
To learn more about the usability and advantages of this concept you can follow up on
:ref:`Customizing Core and Maze Envs <custom_core_maze_envs>`.

In the remainder of this part of the tutorial we will implement the ``Cutting2DEnvironment`` (MazeEnv)
as well as a :ref:`corresponding set of interfaces <env_from_scratch-maze_env-s2o>`.

.. literalinclude:: ../../../../tutorials/tutorial_maze_env/part03_maze_env/env/maze_env.py
  :language: PYTHON
  :caption: env/maze_env.py

The MazeEnv is instantiated with the underlying CoreEnv and the two interfaces for MazeStates and MazeActions.
For convenience we also add a ``maze_env_factory`` to instantiate the MazeEnv from the original environment parameter
set. This will be useful in the next part of the tutorial where we will train an agent based on this environment.

.. _env_from_scratch-maze_env-s2o:

ObservationConversionInterface
------------------------------

The :class:`~maze.core.env.observation_conversion.ObservationConversionInterface`
converts CoreEnv MazeState objects into machine readable Gym-style observations
and defines the respective Gym observation space.
In the present cases the observation is defined as a dictionary with the following structure:

- *inventory*: 2d array representing all pieces currently in inventory
- *inventory_size*: count of pieces currently in inventory
- *order*: 2d vector representing the customer order (current demand)

.. literalinclude:: ../../../../tutorials/tutorial_maze_env/part03_maze_env/space_interfaces/dict_observation_conversion.py
  :language: PYTHON
  :caption: space_interfaces/dict_observation_conversion.py

ActionConversionInterface
-------------------------

The :class:`~maze.core.env.action_conversion.ActionConversionInterface`
converts agent actions into CoreEnv MazeAction objects
and defines the respective Gym action space.
In the present cases the action is defined as a dictionary with the following structure:

- *piece_idx*: id of the inventory piece that should be used for cutting
- *rotation*: defines whether to rotate the piece for cutting or not
- *order*: defines the cutting order (*xy* vs. *yx*)

.. literalinclude:: ../../../../tutorials/tutorial_maze_env/part03_maze_env/space_interfaces/dict_action_conversion.py
  :language: PYTHON
  :caption: space_interfaces/dict_action_conversion.py

Updating the CoreEnv
--------------------

For the sake of completeness we also show two more minor modifications required in the CoreEnv,
which are not too important for this tutorial at the moment.
In short, the :class:`StructuredEnv <maze.core.env.structured_env.StructuredEnv>` interface supports interaction patterns
beyond standard Gym environments to model for example hierarchical or multi-agent RL problems.
We will get back to this in our more advanced tutorials.

The code below defines that the current version of the environment requires
only **one actor** (id 0) with a **single policy** (id 0) that is **never done**.

.. literalinclude:: ../../../../tutorials/tutorial_maze_env/part03_maze_env/env/core_env_snippet.py
  :language: PYTHON
  :caption: env/core_env.py

Test Script
-----------

The following snippet will instantiate the environment and run it for 15 steps.

Note that (compared to the :ref:`previous example <env_from_scratch-core_env-main>`) we are now:

-  working with observations and actions instead of MazeStates and MazeActions
-  able to sample actions from the action_space object

.. literalinclude:: ../../../../tutorials/tutorial_maze_env/part03_maze_env/main.py
  :language: PYTHON
  :caption: main.py

.. code:: bash

    reward -1 | done False | info {'msg': 'valid_cut'}
    reward 0 | done False | info {'msg': 'valid_cut'}
    reward 0 | done False | info {'msg': 'valid_cut'}
    reward 0 | done False | info {'error': 'piece_id_out_of_bounds'}
    reward 0 | done False | info {'error': 'piece_id_out_of_bounds'}
    ...