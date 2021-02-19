.. _env_from_scratch-core_env:

Implementing the CoreEnv
========================

The complete code for this part of the tutorial
`can be found here <https://github.com/enlite-ai/maze/tree/main/tutorials/tutorial_maze_env/part01_core_env>`_


.. code:: bash

    # file structure
    - cutting_2d
        - main.py
        - env
            - core_env.py
            - inventory.py
            - maze_state.py
            - maze_action.py

.. contents:: Page Overview
    :depth: 1
    :local:
    :backlinks: top

.. _env_from_scratch-core_env-code:

CoreEnv
-------

The first component we need to implement is the :ref:`Core Environment <env_hierarchy-core_env>`
which defines the main mechanics and functionality of the environment.

For this example we will call it ``Cutting2DCoreEnvironment``.
As for any other Gym environment we need to implement several methods according to the
:class:`~maze.core.env.core_env.CoreEnv` interface.
We will start with the very basic components and add more and more features (complexity) throughout this tutorial:

- :meth:`~maze.core.env.core_env.CoreEnv.step`: Implements the cutting mechanics.
- :meth:`~maze.core.env.core_env.CoreEnv.reset`: Resets the environment as well as the piece inventory.
- :meth:`~maze.core.env.core_env.CoreEnv.seed`: Sets the random state of the environment for reproducibility.
- :meth:`~maze.core.env.core_env.CoreEnv.close`: Can be used for cleanup.
- :meth:`~maze.core.env.core_env.CoreEnv.get_maze_state`: Returns the current MazeState of the environment.

You can find the implementation of the basic version of the ``Cutting2DCoreEnvironment`` below.

.. literalinclude:: ../../../../tutorials/tutorial_maze_env/part01_core_env/env/core_env.py
  :language: PYTHON
  :caption: env/core_env.py

Environment Components
----------------------

To keep the implementation of the core environment short and clean
we introduces a dedicated ``Inventory`` class providing functionality for:

- maintaining the inventory of available cutting pieces
- replenishing new *raw inventory pieces* if required
- the cutting logic of the environment

.. literalinclude:: ../../../../tutorials/tutorial_maze_env/part01_core_env/env/inventory.py
  :language: PYTHON
  :caption: env/inventory.py

MazeState and MazeAction
------------------------

As motivated and explained in more detail in our tutorial on
:ref:`Customizing Core and Maze Envs <custom_core_maze_envs>` CoreEnvs rely on MazeState and MazeAction objects
for interacting with an agent.

For the present case this is a ``Cutting2DMazeState``

.. literalinclude:: ../../../../tutorials/tutorial_maze_env/part01_core_env/env/maze_state.py
  :language: PYTHON
  :caption: env/maze_state.py

and a ``Cutting2DMazeAction`` defining which inventory piece
to cut in which cutting order and orientation.

.. literalinclude:: ../../../../tutorials/tutorial_maze_env/part01_core_env/env/maze_action.py
  :language: PYTHON
  :caption: env/maze_action.py

These two classes are utilized in the :ref:`CoreEnv code above <env_from_scratch-core_env-code>`.

.. _env_from_scratch-core_env-main:

Test Script
-----------

The following snippet will instantiate the environment and run it for 15 steps.

.. literalinclude:: ../../../../tutorials/tutorial_maze_env/part01_core_env/main.py
  :language: PYTHON
  :caption: main.py

When running the script you should get the following command line output:

.. code:: bash

    reward -1 | done False | info {'msg': 'valid_cut'}
    reward 0 | done False | info {'msg': 'valid_cut'}
    reward 0 | done False | info {'msg': 'valid_cut'}
    ...