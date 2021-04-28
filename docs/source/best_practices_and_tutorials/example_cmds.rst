Rollout and Training Examples
=============================

Run a rollout to test an environment with random action sampling:

.. code-block:: console

    maze-run -cn conf_rollout env.name=CartPole-v1 policy=random_policy

Run a rollout and render the state of the environment:

.. code-block:: console

    maze-run -cn conf_rollout env.name=CartPole-v1 policy=random_policy \
    runner=sequential runner.render=true

Train a policy with evolutionary strategies (ES):

.. code-block:: console

    maze-run -cn conf_train env.name=CartPole-v1 algorithm=es model=vector_obs

Train a policy with with an actor-critic trainer such as A2C:

.. code-block:: console

    maze-run -cn conf_train env.name=CartPole-v1 algorithm=a2c \
    model=vector_obs critic=template_state

Resume training from a previous model state:

.. code-block:: console

    maze-run -cn conf_train env.name=CartPole-v1 algorithm=a2c \
    model=vector_obs critic=template_state input_dir=outputs/<experiment-dir>

Run a rollout of a policy, trained with the command above:

.. code-block:: console

    maze-run -cn conf_rollout env.name=CartPole-v1 model=vector_obs \
    policy=torch_policy input_dir=outputs/<experiment-dir>