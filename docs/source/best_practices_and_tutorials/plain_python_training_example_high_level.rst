.. _end_to_end_python_high_level:

Plain Python Training Example (high-level)
=============================================

This tutorial demonstrates how to train an A2C agent with Maze in plain Python utilizing :class:`~maze.api.run_context.RunContext`. In the process it introduces and explains some of Maze' most important components and concepts, going from a plain vanilla setup to an increasingly customized configuration.

This is complementary to the article on :ref:`low-level training in plain Python <end_to_end_python_low_level>`, which guides through the same setup (but without :class:`~maze.api.run_context.RunContext` support).

Environment Setup
-----------------

We will first prepare our environment for use with Maze. In order to use Maze's parallelization capabilities, it is necessary to define a factory function that returns a :class:`~maze.core.env.maze_env.MazeEnv` of your environment. This is easily done for Gym environments:

.. code-block:: python

    def cartpole_env_factory():
        """ Env factory for the cartpole MazeEnv """
        # Registered gym environments can be instantiated first and then provided to GymMazeEnv:
        cartpole_env = gym.make("CartPole-v0")
        maze_env = GymMazeEnv(env=cartpole_env)

        # Another possibility is to supply the gym env string to GymMazeEnv directly:
        maze_env = GymMazeEnv(env="CartPole-v0")

        return maze_env

    env = cartpole_env_factory()

If you have your own environment (that is not a :code:`gym.Env`) you must transform it into a MazeEnv yourself, as is shown :ref:`here <env_from_scratch-maze_env>`, and have your factory return that. If it is a custom gym env it can be instantiated with our wrapper as shown above.

Algorithm Setup
-----------------

We use A2C for this example. The algorithm_config for A2C can be found :ref:`here <maze_trainers-a2c>`. The hyperparameters will be supplied to Maze with an algorithm-dependent AlgorithmConfig object. The one for A2C is :class:`~maze.train.trainers.a2c.a2c_algorithm_config.A2CAlgorithmConfig`. We will use the default parameters, which can also be found :ref:`here <maze_trainers-a2c>`.

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
        device='cpu'
   )

----

Having defined our environment and configured our algorithm we're ready to train:

.. code-block:: python

    rc = maze.api.run_context.RunContext(env=cartpole_env_factory, algorithm=algorithm_config)
    rc.train()

Custom Model Setup
------------------

However, it can be advisable to create customized networks taking full advantage of the available data. For this reason Maze supports plugging in customized policy and value networks.

Our goal is to hence train an agent with A2C using customized policy and critic networks:

.. code-block:: python

    rc = maze.api.run_context.RunContext(
        env=cartpole_env_factory,
        algorithm=algorithm_config,
        policy=...,
        critic=...
    )
    rc.train()


Here we will pay special attention to emphasize the format required by Maze. When creating your own models, it is important to know three things:

#. Maze works with dictionaries throughout, which means that arguments for the constructor and the input and return values of the forward method are dicts with user-defined keys. In a nutshell, instances of :class:`MazeEnv<maze.core.env.maze_env.MazeEnv>` can have different *steps* indicating the currently active task. Each step is associated with a policy, so an environment with different steps can also have different policy. By default environments have only step *0*. The required format for models is explained in more detail :ref:`here <custom_models_signature>`.
#. Policy networks and value network constructors have required arguments: for policy nets, these are `obs_shapes` and `action_logit_dicts`, for value nets, this is `obs_shapes`.
#. Policies and critics are not passed directly, but via composer objects - i.e. classes of type :class:`~maze.perception.models.policies.base_policy_composer.BasePolicyComposer` or :class:`~maze.perception.models.critics.critic_composer_interface.CriticComposerInterface`, respectively. Such composer classes are able to generate policy instances.

**Policy Customization**

To instantiate e.g. a :class:`ProbabilisticPolicyComposer<maze.perception.models.policies.probabilistic_policy_composer.ProbabilisticPolicyComposer>`, we require the following arguments:

#. The policy network.
#. A specification of the probability distribution as an instance of :class:`~maze.distributions.distribution_mapper.DistributionMapper`.
#. Dictionaries describing the action and observation spaces.
#. The numbers of agents active in the corresponding steps.
#. The IDs of substeps in which agents do not share the same networks.

**Policy Network.** First, let us create the latter as a simple linear mapping network with the required constraints:

.. code-block:: python

    class CartpolePolicyNet(nn.Module):
        """ Simple linear policy net for demonstration purposes. """
        def __init__(self, obs_shapes: Dict[str, Sequence[int]], action_logit_shapes: Dict[str, Sequence[int]]):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(
                    in_features=obs_shapes['observation'][0],
                    out_features=action_logit_shapes['action'][0]
                )
            )

        def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            # Since x_dict has to be a dictionary in Maze, we extract the input for the network.
            x = x_dict['observation']

            # Do the forward pass.
            logits = self.net(x)

            # Since the return value has to be a dict again, put the
            # forward pass result into a dict with the correct key.
            logits_dict = {'action': logits}

            return logits_dict

    # Instantiate our custom policy net.
    policy_net = CartpolePolicyNet(
        obs_shapes={'observation': env.observation_space.spaces['observation'].shape},
        action_logit_shapes={'action': (env.action_space.spaces['action'].n,)}
    )

Optionally, we can wrap our policy network with a :class:`~maze.perception.blocks.general.torch_model_block.TorchModelBlock`, which applies shape normalization (see :class:`~maze.perception.blocks.shape_normalization.ShapeNormalizationBlock`):

.. code-block:: python

    policy_net = TorchModelBlock(
        in_keys='observation',
        out_keys='action',
        in_shapes=env.observation_space.spaces['observation'].shape,
        in_num_dims=[2],
        out_num_dims=2,
        net=policy_net
    )

Since Maze offers the capability of supporting multiple actors, we need to map each :code:`policy_net` to its corresponding actor ID. As we have only one, this mapping is trivial:

.. code-block:: python

    policy_networks = [policy_net]  # Alternative: {0: policy_net}

**Policy Distribution.** Initializing the proper probability distribution for the policy is rather easy with Maze.
Simply provide the :code:`~maze.distributions.distribution_mapper.DistributionMapper` with the environment's action space and you automatically get the proper distribution to use.

.. code-block:: python

    distribution_mapper = DistributionMapper(action_space=env.action_space, distribution_mapper_config={})

Optionally, you can specify a different distribution with the :code:`distribution_mapper_config` argument. Using a
:class:`~maze.distributions.categorical.CategoricalProbabilityDistribution` for a discrete action space would be done with

.. code-block:: python

    distribution_mapper = DistributionMapper(
        action_space=action_space,
        distribution_mapper_config=[{
            "action_space": gym.spaces.Discrete,
            "distribution": "maze.distributions.categorical.CategoricalProbabilityDistribution"}])

Since the standard distribution taken by Maze for a discrete action space is a Categorical distribution anyway (as can be seen :ref:`here <action_spaces_and_distributions>`), both definitions of :code:`distribution_mapper` have the same result. For more information about the DistributionMapper, see :ref:`Action Spaces and Distributions <action_spaces_and_distributions_module>`.

----

**Policy Composer.** The remaining arguments (action and observation space dictionaries, numbers of agents per step, ID of substeps with non-shared networks) are trivial in our case, as they can easily be derived from an instance of our environment.
We can thus now set up a policy composer with our custom policy:

.. code-block:: python

    policy_composer = ProbabilisticPolicyComposer(
        action_spaces_dict=env.action_spaces_dict,
        observation_spaces_dict=env.observation_spaces_dict,
        distribution_mapper=distribution_mapper,
        networks=policy_networks,
        # We have only one agent and network, thus this is an empty list.
        substeps_with_separate_agent_nets=[],
        # We have only one step and one agent.
        agent_counts_dict={0: 1}
    )

Once we have our policy composer, we are ready to train.

.. code-block:: python

    rc = maze.api.run_context.RunContext(
        env=cartpole_env_factory,
        algorithm=algorithm_config,
        policy=policy_composer
    )
    rc.train()

**Critic Customization**

Customizing the critic can be done quite similarly to the policy customization, the main difference being that we do not need a probability distribution.

First we define our value network.

.. code-block:: python

    class CartpoleValueNet(nn.Module):
        """ Simple linear value net for demonstration purposes. """
        def __init__(self, obs_shapes: Dict[str, Sequence[int]]):
            super().__init__()
            self.value_net = nn.Sequential(nn.Linear(in_features=obs_shapes['observation'][0], out_features=1))



        def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """ Forward method. """
            # The same as for the policy can be said about the value
            # net: Inputs and outputs have to be dicts.
            x = x_dict['observation']

            value = self.value_net(x)

            value_dict = {'value': value}
            return value_dict

We instantiate our policy network and wrap it in a :class:`TorchModelBlock<maze.perception.blocks.general.torch_model_block.TorchModelBlock>` as done for the policy network.

.. code-block:: python

    value_networks = {
        0: TorchModelBlock(
            in_keys='observation', out_keys='value',
            in_shapes=observation_space.spaces['observation'].shape,
            in_num_dims=[2],
            out_num_dims=2,
            net=CartpoleValueNet(obs_shapes=env.observation_space.spaces['observation'].shape)
        )
    }

**Instantiating the Critic.** This step is analogous to the instantiation of the policy above. In Maze, critics can have different forms (see :ref:`Value Functions (Critics) <critics_section>`).
Here, we use a simple shared critic. Shared means that the same critic will be used for all sub-steps (in a multi-step
setting) and all actors.
Since we only have one actor in this example and are in a one-step setting, the :class:`~maze.core.agent.torch_state_critic.TorchSharedStateCritic` reduces to
a vanilla :class:`~maze.core.agent.state_critic.StateCritic` (aka a state-dependent value function).

.. code-block:: python

    critic_composer = SharedStateCriticComposer(
        observation_spaces_dict=env.observation_spaces_dict,
        agent_counts_dict={0: 1},
        networks=value_networks,
        stack_observations=True
    )

**Training**

Having instantiated customized policy and critic composers we can train our model:

.. code-block:: python

    rc = run_context.RunContext(
        env=cartpole_env_factory,
        algorithm=algorithm_config,
        policy=policy_composer,
        critic=critic_composer
    )
    rc.train()

**Distributed Training**

If we want to train in a distributed manner, it is sufficient to pick the appropriate runner. For now, we might want to parallelize by distributing our environments over several processes. This can be done by utilizing *local* runners, whose utilization is straightforward:

.. code-block:: python

    rc = run_context.RunContext(
        env=cartpole_env_factory,
        algorithm=algorithm_config,
        policy=policy_composer,
        critic=critic_composer,
        runner="local"
    )
    rc.train(n_epochs=1)

**Evaluation**

We can evaluate our performance with a :class:`~maze.train.trainers.common.evaluators.rollout_evaluator.RolloutEvaluator`. In order for this to work with our environment, we wrap it with a :class:`~maze.core.wrappers.log_stats_wrapper.LogStatsWrapper` to ensure it has the logging capabilities required by the :class:`~maze.train.trainers.common.evaluators.rollout_evaluator.RolloutEvaluator`.

.. code-block:: python

    evaluator = RolloutEvaluator(
        eval_env=LogStatsWrapper.wrap(cartpole_env_factory(), logging_prefix="eval"),
        n_episodes=3,
        model_selection=None
    )
    evaluator.evaluate(rc.policy)

Full Python Code
----------------

Here is the code without documentation for easier copy-pasting:

.. literalinclude:: code_snippets/plain_python_training_high_level.py
  :language: PYTHON
