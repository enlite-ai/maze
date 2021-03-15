.. _end_to_end_python:

Plain Python Training Example
=============================
We will demonstrate how to do a manual RL training loop with Maze in plain Python, i.e., without any Hydra calls.
This example serves to demonstrate how to transform your own custom RL loop to Maze. Let's assume you want to train an
A2C agent for the Cartpole environment and let's also assume that you have already spent much time in creating your own
networks for this environment, i.e., you want to use your own networks but want to transition to Maze.

Environment Setup
-----------------

We will first prepare our environment for use with Maze. In order to use Maze's parallelization capabilities, it
is necessary to define a factory function that returns a MazeEnv of your environment. This is easily done for
Gym environments:

.. code-block:: python

    def cartpole_env_factory():
        """ Env factory for the cartpole MazeEnv """
        # Registered gym environments can be instantiated first and then provided to GymMazeEnv:
        cartpole_env = gym.make("CartPole-v0")
        maze_env = GymMazeEnv(env=cartpole_env)

        # Another possibility is to supply the gym env string to GymMazeEnv directly:
        maze_env = GymMazeEnv(env="CartPole-v0")

        return maze_env

If you have your own environment you must transform it into a MazeEnv yourself, as is shown
:ref:`here <env_from_scratch-maze_env>`, and have your factory return that.

We instantiate one environment. This will be used for convenient access to observation and action spaces later.

.. code-block:: python

    env = cartpole_env_factory()
    observation_space = env.observation_space
    action_space = env.action_space



Model Setup
-----------
Now that the environment setup is done, let us define the policy and value networks that will be used. We will not
re-use the networks that were introduced in
:ref:`Example 3: Custom Networks with (plain PyTorch) Python <custom_example_3>`
as they already adhere to the Maze model interface. Here, we would like to show how to transform any models that
you already have to the necessary Maze interface.

Assume that you have created the following policy and
value networks for the cartpole environment:

.. code-block:: python

    class CartpolePolicyNet(nn.Module):
        """ Simple linear policy net for demonstration purposes """
        def __init__(self, in_features, out_features):
            super(CartpolePolicyNet, self).__init__()
            self.dense = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features))

        def forward(self, x):
            """ Forward method """
            return self.dense(x)


and

.. code-block:: python

    class CartpoleValueNet(nn.Module):
        """ Simple linear value net for demonstration purposes """
        def __init__(self, in_features):
            super(CartpoleValueNet, self).__init__()
            self.dense = nn.Sequential(nn.Linear(in_features=in_features, out_features=1))

        def forward(self, x):
            """ Forward method """
            return self.dense(x)

The first step will be to transform these models into a form that Maze can understand. It is important to know that Maze
works with dictionaries, which means that parameter and return values of the forward method are dicts with
user-defined keys. Another requirement are the parameters for the model initialization, as is explained
:ref:`here <custom_models_signature>`
: required arguments for the
policy nets are the arguments `obs_shapes` and `action_logit_dicts`. The value net is required to have the
argument `obs_shapes`.
A transformation of the present networks to networks with the required form can be easily achieved by
wrapping the models:

.. code-block:: python

    class WrappedCartpolePolicyNet(nn.Module):
        """ Wrapper for a model that transforms the network into a Maze. compatible one. """
        def __init__(self, obs_shapes, action_logit_shapes):
            super(WrappedCartpolePolicyNet, self).__init__()
            self.policy_network = CartpolePolicyNet(in_features=obs_shapes[0], out_features=action_logit_shapes[0])

        def forward(self, x_dict):
            logits_dict = {'action': self.policy_network.forward(x_dict['observation'])}
            return logits_dict

and

.. code-block:: python

    class WrappedCartpoleValueNet(nn.Module):
        """ Wrapper for a model that transforms the network into a Maze. compatible one. """
        def __init__(self, obs_shapes):
            super(WrappedCartpoleValueNet, self).__init__()
            self.value_net = CartpoleValueNet(in_features=obs_shapes[0])

        def forward(self, x_dict):
            """ Forward method. """
            value_dict = {'value': self.value_net.forward(x_dict['observation'])}
            return value_dict


**Policy Setup**

For a policy, we need a parametrization for the policy (provided by the policy network) and a probability distribution
we can sample from. We will subsequently define and instantiate each of these.

**Policy Network**

Instantiate a policy with the correct shapes of observation and action spaces.

.. code-block:: python

    policy_net = WrappedCartpolePolicyNet(
        obs_shapes=observation_space.spaces['observation'].shape,
        action_logit_shapes=(action_space.spaces['action'].n,))


We can use one of Mazes capabilities, shape normalization
(see :class:`ShapeNormalizationBlock <maze.perception.blocks.shape_normalization.ShapeNormalizationBlock>`),
with these models by wrapping them with the TorchModelBlock.

.. code-block:: python

    maze_wrapped_policy_net = TorchModelBlock(
        in_keys='observation', out_keys='action',
        in_shapes=observation_space.spaces['observation'].shape, in_num_dims=[2],
        out_num_dims=2, net=policy_net)

Since Maze offers the capability of supporting multiple actors, we need to map each policy_net to its corresponding
actor ID. As we have only one policy, this is a trivial mapping:

.. code-block:: python

    policy_networks = {0: maze_wrapped_policy_net}

**Policy Distribution**

Initializing the proper probability distribution for the policy is rather easy with Maze.
Simply provide the DistributionMapper
with the action space and you automatically get the proper distribution to use.

.. code-block:: python

    distribution_mapper = DistributionMapper(action_space=action_space, distribution_mapper_config={})

Optionally, you can specify a different distribution with the `distribution_mapper_config` argument. Using a
Categorical distribution for a discrete action space would be done with

.. code-block:: python

    distribution_mapper = DistributionMapper(
        action_space=action_space,
        distribution_mapper_config=[{
            "action_space": gym.spaces.Discrete,
            "distribution": "maze.distributions.categorical.CategoricalProbabilityDistribution"}])

Since the standard distribution taken by Maze for a discrete action space is a Categorical distribution anyway
(as can be seen :ref:`here <action_spaces_and_distributions>`),
both definitions of the distribution_mapper have the same result. For more information about the DistributionMapper,
see :ref:`Action Spaces and Distributions <action_spaces_and_distributions_module>`.

**Instantiating the Policy**

We have both necessary ingredients to define a policy: a parametrization, given by the policy network, and a
distribution. With these, we can instantiate a policy. This is done with the TorchPolicy class:

.. code-block:: python

    torch_policy = TorchPolicy(networks=policy_networks,
                               distribution_mapper=distribution_mapper,
                               device='cpu')


**Critic Setup**

The setup of a critic (or value function) is similar to the setup of a policy, the main difference being that
we do not need a probability distribution.

**Value Network**

.. code-block:: python

    value_net = WrappedCartpoleValueNet(obs_shapes=observation_space.spaces['observation'].shape)

    maze_wrapped_value_net = TorchModelBlock(
        in_keys='observation', out_keys='value',
        in_shapes=observation_space.spaces['observation'].shape, in_num_dims=[2],
        out_num_dims=2, net=value_net)

    value_networks = {0: maze_wrapped_value_net}

**Instantiating the Critic**

This step is analogous to the instantiation of the policy above. In Maze, critics can have
different forms (see :ref:`Value Functions (Critics) <critics_section>`).
Here, we use a simple shared critic. Shared means that the same critic will be used for all sub-steps (in a multi-step
setting) and all actors.
Since we only have one actor in this example and are in a one-step setting, the TorchSharedStateCritic reduces to
a vanilla StateCritic (aka a state-dependent value function).

.. code-block:: python

    torch_critic = TorchSharedStateCritic(networks=value_networks, num_policies=1, device='cpu')


**Initializing the ActorCritic Model.**

In Maze, policies and critics are encapsulated by an ActorCritic model. Details
about this can be found in :ref:`Actor-Critics <actor_critics_section>`.
We will use A2C to train the cartpole env. The correct ActorCritic model to use for A2C is the TorchActorCritic:

.. code-block:: python

    actor_critic_model = TorchActorCritic(policy=torch_policy, critic=torch_critic, device='cpu')

Trainer Setup
-------------
The last steps will be the instantiations of the algorithm and corresponding trainer.
We use A2C for this example. The algorithm_config for A2C can be found :ref:`here <maze_trainers-a2c>`.
The hyperparameters will be supplied to Maze with an algorithm-dependent AlgorithmConfig object. The one
for A2C is A2CAlgorithmConfig. We will use the default parameters, which can also be found
:ref:`here <maze_trainers-a2c>`.

.. code-block:: python

    algorithm_config = A2CAlgorithmConfig(
        n_epochs=5,
        epoch_length=25,
        deterministic_eval=False,
        eval_repeats=2,
        patience=15,
        critic_burn_in_epochs=0,
        n_rollout_steps=100,
        lr=0.0005,
        gamma=0.98,
        gae_lambda=1.0,
        policy_loss_coef=1.0,
        value_loss_coef=0.5,
        entropy_coef=0.00025,
        max_grad_norm=0.0,
        device='cpu')

In order to use the distributed trainers, the previously created env factory is supplied to one of Maze's
distribution classes:

.. code-block:: python

    train_envs = DummyStructuredDistributedEnv(
        [cartpole_env_factory for _ in range(2)], logging_prefix="train")
    eval_envs = DummyStructuredDistributedEnv(
        [cartpole_env_factory for _ in range(2)], logging_prefix="eval")

With this, the trainer can be instantiated:

.. code-block:: python

    a2c_trainer = MultiStepA2C(env=train_envs, eval_env=eval_envs,
        algorithm_config=algorithm_config, model=actor_critic_model, model_selection=None)


Train the Agent
---------------
Before starting the training, we will enable logging by calling

.. code-block:: python

    log_dir = '.'
    setup_logging(job_config=None, log_dir=log_dir)

Now, we can train the agent.

.. code-block:: python

    a2c_trainer.train()

To get an out-of sample estimate of our performance, evaluate on the evaluation envs:

.. code-block:: python

    a2c_trainer.evaluate(deterministic=False, repeats=1)


Full Python Code
----------------
Here is the code without documentation for easier copy-pasting:

.. literalinclude:: code_snippets/plain_python_training.py
  :language: PYTHON
