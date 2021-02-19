.. _tutorial_gym_env:

Integrating an Existing Gym Environment
=======================================

Maze supports a seamless integration of
existing `OpenAI Gym environments <https://gym.openai.com/docs/>`_.
To get full Maze feature support for Gym environments we first have to transform them into Maze environments.
This page shows how this is easily accomplished via the :ref:`GymMazeEnv <env_wrappers_ref-gym_env>`.

.. image:: gym_env_wrapper.png
    :width: 60 %
    :align: center

A Gym environment is transformed into a
:class:`GymMazeEnv <maze.core.wrappers.maze_gym_env_wrapper.GymMazeEnv>` by:

 - Wrapping the Gym environment into a
   :class:`GymCoreEnv <maze.core.wrappers.maze_gym_env_wrapper.GymCoreEnv>`.
 - This requires transforming the observation and action spaces into a dictionary spaces via the
   :class:`GymObservationConversion <maze.core.wrappers.maze_gym_env_wrapper.GymObservationConversion>` and
   :class:`GymActionConversion <maze.core.wrappers.maze_gym_env_wrapper.GymActionConversion>`
   interfaces.
 - Finally, the GymCoreEnv is packed into a :class:`GymMazeEnv <maze.core.wrappers.maze_gym_env_wrapper.GymMazeEnv>`
   which is fully compatible with all other Maze components and modules.

To get a better understanding of the overall structure please refer to the
:ref:`Maze environment hierarchy <env-hierarchy>`.

Instantiating a Gym Environment as a Maze Environment
-----------------------------------------------------

The config snippet below shows how to instantiate an existing, already registered Gym environment
as a GymMazeEnv referenced by its environment name (here *CartPole-v0*).

.. code-block:: YAML

    # @package env
    type: maze.core.wrappers.maze_gym_env_wrapper.make_gym_maze_env
    name: "CartPole-v0"

To achieve the same result directly with plain Python you can start with the code snippet below.

.. code-block:: PYTHON

    from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
    env = GymMazeEnv(env="CartPole-v0")

In case your environment is not yet registered with Gym you can also directly instantiate the Gym environment before
passing it to the *GymMazeEnv*.
This might be useful in case you already have your own custom Gym environments implemented.

.. code-block:: PYTHON

    import gym
    from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
    gym_env = gym.make("CartPole-v0")
    env = GymMazeEnv(env=gym_env)

Where to Go Next
----------------

- For further details please see the :ref:`reference documentation <env_wrappers_ref-gym_env>`.
- Next you might be interested in how to :ref:`train an agent for your environment <training>`.
- You might also want to read up on the :ref:`Maze environment hierarchy <env-hierarchy>`
  for the bigger picture.