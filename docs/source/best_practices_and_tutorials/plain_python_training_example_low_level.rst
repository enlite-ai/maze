.. _end_to_end_python_low_level:

Plain Python Training Example (low-level)
=========================================

This tutorial demonstrates how to train an A2C agent with Maze in plain Python without utilizing the maze CLI. In the process it introduces and explains some of Maze' most important components and concepts.

Environment Setup
-----------------

We will first prepare our environment for use with Maze. In order to use Maze's parallelization capabilities, it is necessary to define a factory function that returns a MazeEnv of your environment. This is easily done for
Gym environments:

.. code-block:: python

    def cartpole_env_factory():
        """ Env factory for the cartpole MazeEnv """
        # Registered gym environments can be instantiated first and then provided to GymMazeEnv:
        cartpole_env = gym.make("CartPole-v1")
        maze_env = GymMazeEnv(env=cartpole_env)

        # Another possibility is to supply the gym env string to GymMazeEnv directly:
        maze_env = GymMazeEnv(env="CartPole-v1")

        return maze_env

If you have your own environment (that is not a :code:`gym.Env`) you must transform it into a MazeEnv yourself, as is shown :ref:`here <env_from_scratch-maze_env>`, and have your factory return that. If it is a custom gym env it can be instantiated with our wrapper as shown above.

We instantiate one environment. This will be used for convenient access to observation and action spaces later.

.. code-block:: python

    env = cartpole_env_factory()
    observation_space = env.observation_space
    action_space = env.action_space



Model Setup
-----------
Now that the environment setup is done, let us develop the policy and value networks that will be used. We will
pay special attention to emphasize the format required by Maze. When creating your own models,
it is important to know two things:

#. Maze works with dictionaries throughout, which means that arguments for the constructor and the input and return values of the forward  method are dicts with user-defined keys.
#. Policy networks and value network constructors have required arguments: for policy nets, these are `obs_shapes` and `action_logit_dicts`, for value nets, this is `obs_shapes`.

The required format is explained in more detail :ref:`here <custom_models_signature>`. With this in mind, let us
create a simple linear mapping network with the required constraints:

.. code-block:: python

    class CartpolePolicyNet(nn.Module):
        """ Simple linear policy net for demonstration purposes. """
        def __init__(self, obs_shapes: Sequence[int], action_logit_shapes: Sequence[int]):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features=obs_shapes[0], out_features=action_logit_shapes[0])
            )

        def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            # Since x_dict has to be a dictionary in Maze, we extract the input for the network.
            x = x_dict['observation']

            # Do the forward pass.
            logits = self.net(x)

            # Since the return value has to be a dict again, put the
            # forward pass result into a dict with the  correct key.
            logits_dict = {'action': logits}

            return logits_dict

    # Instantiate our custom policy net.
    policy_net = CartpolePolicyNet(
        obs_shapes=env.observation_space.spaces['observation'].shape,
        action_logit_shapes=(env.action_space.spaces['action'].n,)
    )

and

.. code-block:: python

    class CartpoleValueNet(nn.Module):
        """ Simple linear value net for demonstration purposes. """
        def __init__(self, obs_shapes: Sequence[int]):
            super().__init__()
            self.value_net = nn.Sequential(nn.Linear(in_features=obs_shapes[0], out_features=1))


        def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """ Forward method. """
            # The same as for the policy can be said about the value
            # net: Inputs and outputs have to be dicts.
            x = x_dict['observation']

            value = self.value_net(x)

            value_dict = {'value': value}
            return value_dict

**Policy Setup**

For a policy, we need a parametrization for the policy (provided by the policy network) and a probability distribution
we can sample from. We will subsequently define and instantiate each of these.

**Policy Network**

Instantiate a policy with the correct shapes of observation and action spaces.

.. code-block:: python

    policy_net = CartpolePolicyNet(
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

    value_net = CartpoleValueNet(obs_shapes=observation_space.spaces['observation'].shape)

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
        device='cpu',
        rollout_evaluator=RolloutEvaluator(
            eval_env=SequentialVectorEnv([cartpole_env_factory]),
            n_episodes=1,
            model_selection=None,
            deterministic=True
        )
    )

In order to use the distributed trainers, we create a vector environment (i.e., multiple environment
instances encapsulated to be stepped simultaneously) using the environment factory function:

.. code-block:: python

    train_envs = SequentialVectorEnv(
        [cartpole_env_factory for _ in range(2)], logging_prefix="train")
    eval_envs = SequentialVectorEnv(
        [cartpole_env_factory for _ in range(2)], logging_prefix="eval")

(In this case, we create sequential vector environments, i.e. all environment instances are
located in the main process and stepped sequentially.
When we are ready to scale the training, we might want to use e.g. sub-process distributed vector environments.)

For this example, we want to save the parameters of the best model in terms of mean achieved reward. This is done
with the
:class:`BestModelSelection <maze.train.trainers.common.model_selection.best_model_selection.BestModelSelection>` class,
an instance of which will be provided to the trainer.

.. code-block:: python

    model_selection = BestModelSelection(dump_file="params.pt", model=actor_critic_model)

We can now instantiate an A2C trainer:

.. code-block:: python

    a2c_trainer = A2C(
        env=train_envs,
        algorithm_config=algorithm_config,
        model=actor_critic_model,
        model_selection=model_selection,
        evaluator=algorithm_config.rollout_evaluator
    )


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

.. literalinclude:: code_snippets/plain_python_training_low_level.py
  :language: PYTHON
