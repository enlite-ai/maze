.. |stable_baselines_3_read_the_docs| raw:: html

   <a href="https://stable-baselines3.readthedocs.io/en/master/index.html" target="_blank">stable-baselines3</a>

.. |stable_baselines_3_quickstart| raw:: html

   <a href="https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html" target="_blank">getting started<a>

.. |stable_baselines_3_installation| raw:: html

   <a href="https://stable-baselines3.readthedocs.io/en/master/guide/install.html" target="_blank">install stable-baselines3</a>

.. _maze_and_others:

Combining Maze with other RL Frameworks
=======================================
This tutorial explains how to use general Maze features in combination with existing RL frameworks.
In particular, we will apply :ref:`observation normalization <observation_normalization>` before optimizing a policy
with the |stable_baselines_3_read_the_docs| A2C trainer.
When adding new features to Maze we put a strong emphasis on reusablity
to allow you to make use of as much of these features as possible
but still give you the opportunity to stick to the optimization framework you are most comfortable or familiar with.

We rely on |stable_baselines_3_read_the_docs| for this tutorial.
However, it is important to note that the examples below will also work with any other Python-based RL framework
compatible with Gym environments.

We provide two different versions showing how to arrive at an observation normalized environment.
The first one is written in :ref:`plain Python <maze_and_others-reuse_python>`
where the second reproduces the Python example with a :ref:`Hydra configuration <maze_and_others-reuse_hydra>`.

.. note::

    Although, this tutorial explains how to reuse observation normalization there is of course no
    limitation to this sole feature. So if you find this useful we definitely recommend you to browse through our
    *Environment Customization* section in the sidebar.

.. _maze_and_others-reuse_python:

Reusing Environment Customization Features
------------------------------------------

The basis for this tutorial is the official |stable_baselines_3_quickstart|
snippet of stable-baselines showing how to train and run A2C on a CartPole environment.
We added a few comments to make things a bit more explicit.

If you would like to run this example yourself make sure to |stable_baselines_3_installation| first.

.. literalinclude:: code_snippets/stable_baselines_getting_started.py
  :language: python

Below you find exactly the same example but with an observation normalized environment.
The following modifications compared to the example above are required:

 - Instantiate a GymMazeEnv instead of a standard Gym environment
 - Wrap the environment with the ObservationNormalizationWrapper
 - Estimate normalization statistics from actual environment interactions

As you might already have experienced, re-coding these steps for different environments and experiments
can get quite cumbersome.
The wrapper also dumps the estimated statistics in a file (*statistics.pkl*)
to reuse them later on for agent deployment.

.. literalinclude:: code_snippets/maze_and_stable_baselines.py
  :language: python

.. _maze_and_others-reuse_hydra:

Reusing the Hydra Configuration System
--------------------------------------

This example is identical to the the previous one but instead of instantiated everything
:ref:`directly from Python <maze_and_others-reuse_python>` it utilizes the :ref:`Hydra configuration system <hydra>`.

.. literalinclude:: code_snippets/hydra_and_stable_baselines.py
  :language: python

This is the corresponding hydra config:

.. literalinclude:: code_snippets/conf/hydra_config.yaml
  :language: yaml

Where to Go Next
----------------

 - You can learn more about the :ref:`Hydra configuration system <hydra>`.
 - As :ref:`observation normalization <observation_normalization>` is not the scope of this section we recommend to
   read up on this in the dedicated section.
 - You might be also interested in :ref:`observation pre-processing <observation_pre_processing>` and the remaining
   environment customization options (see sidebar *Environment Customization*).
 - You can also check out the built-in Maze Trainers with full dictionary space support for observations and actions.
 - You can also make use of the full :ref:`Maze environment hierarchy <env-hierarchy>`.