.. _run_context:

High-level API: RunContext
==========================

This page describes the RunContext, a high-level API for training, rollout and evaluation of Maze agents in plain Python.

Motivation
----------

Maze :ref:`utilizes Hydra <hydra>` to facilitate a powerful configuration mechanism boosting developers' flexibility in their creation of reinforcement learning projects. Hydra however is geared towards command line usage, so these benefits are not accessible when working with individual Maze components (like :class:`~maze.train.trainers.common.trainer.Trainer`, :class:`~maze.core.wrappers.wrapper.Wrapper`, :class:`~maze.core.env.maze_env.MazeEnv`, ...) and composing them manually in Python.

E.g.: It is not possible to generate components directly from the provided configuration modules. This would however be quite useful, as it allows to loads pre-configured (sets of) components. This can be exemplified by the `pixel_obs` wrapper configuration module, which defines several wrappers useful for the preprocessing, normalization and logging of such pixel space observations. Via the CLI this can be loaded trivially via :code:`... wrappers=pixel_obs ...` - yet there is no obvious way to leverage Maze' Hydra-based configuration system from within a Python script. This also affects other features, like the inability to instantiate objects from a YAML-/dict-based configuration object (which can be very convienient with increasing experiment or application complexity).

This motivates the introduction of :class:`~maze.api.run_context.RunContext`, a high-level API for training, rollout and evaluation. When working with Maze from within a Python script (as opposed to via the CLI with :code:`maze-run`) we highly recommend that you start with :class:`~maze.api.run_context.RunContext`: It requires very little configuration overhead to get things rolling, yet offers a lot of flexibility if you require additional configuration. While there might be cases where this is not sufficient, we expect that this would not happen too frequently.

Comparison with the CLI (maze-run)
----------------------------------

We designed :class:`~maze.api.run_context.RunContext` to be largely congruent with the CLI, i.e. :code:`maze-run`. It utilizes Hydra internally and offers the same base functionality, but differs in a couple of ways - :class:`~maze.api.run_context.RunContext` ...

* ... is a recent addition and still lacks support for a number of capabilities: Rolling out and evaluating a policy is not fully supported yet, as are RLlib integration and some of the more advanced Hydra features like multi-runs or experiments. These issues (particularly rollout and evaluation support) are on our todo list however and will be implemented shortly.
* ... accepts (most) components to be specified as instantiated complex Python objects, configuration dictionaries or configuration module name. In contrast, the CLI accepts the specification of components as configuration module name or as primitive values. As of now this entails however that once instantiated Python objects are passed, the customary experiment configuration cannot be logged anymore due to a lack of knowledge about the corresponding configuration dictionary. This issue is on our roadmap.
* ... offers a few additional options for convenience' sake, such as output suppression via :code:`silent=True` or setting the working directory via :code:`run_dir='...'`.

Usage
-----

This section aims to convey the principal ideas and features of :class:`~maze.api.run_context.RunContext`. For further explanation and a detailled discussion of the exposed interface as well as auxiliary components and utilities see :ref:`here <run_context_ref>`.

**Initialization**

As mentioned previously, the :class:`~maze.api.run_context.RunContext` API is largely congruent with the :code:`maze-run` CLI. Consequently the initialization can be done in a similar fashion. Here is one example with a particular training run configuration using the CLI, :class:`~maze.api.run_context.RunContext` initialized with configuration module names and :class:`~maze.api.run_context.RunContext` initialized with a mix of configuration module names and complex Python objects.

.. tabs::

    .. code-tab:: console

        maze-run -cn conf_train env.name=CartPole-v0 algorithm=a2c model=vector_obs critic=template_state

    .. code-tab:: python API, CLI-style initialization

        rc = RunContext(
            algorithm="a2c",
            overrides={"env.name": "CartPole-v0"},
            model="vector_obs",
            critic="template_state"
        )

    .. code-tab:: python API, mixed initialization

        alg_config = A2CAlgorithmConfig(
            n_epochs=1,
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

        rc = RunContext(
            algorithm=alg_config,
            overrides={"env.name": "CartPole-v0"},
            model="vector_obs",
            critic="template_state"
        )

Environments cannot be passed in instantiated form, but instead as callable environment factories:

.. code-block:: python

        rc = RunContext(env=lambda: GymMazeEnv('CartPole-v0'))

As with the CLI, any attribute in the configuration hierarchy can be overridden, not just the explicitly exposed top-level attributes like :code:`env` or :code:`algorithm`. This can be achieved using the :code:`overrides` dictionary as seen above for :code:`"env.name"`. It is also possible to pass complex values:

 .. tabs::

    .. code-tab:: python With a configuration dictionary

        policy_composer_config = {
            '_target_': 'maze.perception.models.policies.ProbabilisticPolicyComposer',
            'networks': [{
                '_target_': 'maze.perception.models.built_in.flatten_concat.FlattenConcatPolicyNet',
                'non_lin': 'torch.nn.Tanh',
                'hidden_units': [256, 256]
            }],
            "substeps_with_separate_agent_nets": [],
            "agent_counts_dict": {0: 1}
        }
        rc = RunContext(overrides={"model.policy": policy_composer_config})

    .. code-tab:: python With an instantiated object

        policy_composer = ProbabilisticPolicyComposer(
                action_spaces_dict=env.action_spaces_dict,
                observation_spaces_dict=env.observation_spaces_dict,
                distribution_mapper=DistributionMapper(action_space=env.action_space, distribution_mapper_config={}),
                networks=[{
                    '_target_': 'maze.perception.models.built_in.flatten_concat.FlattenConcatPolicyNet',
                    'non_lin': 'torch.nn.Tanh',
                    'hidden_units': [222, 222]
                }],
                substeps_with_separate_agent_nets=[],
                agent_counts_dict={0: 1}
        )
        rc = RunContext(overrides={"model.policy": policy_composer})

Note that by design configuration module name resolution is not triggered for attributes in :code:`overrides`. This is necessary for some of the explicitly exposed arguments however. We recommend *strongly* to pass an argument explicitly, if it is explicitly exposed - otherwise a correct assembly of the underlying configuration structure cannot be guaranteed. E.g. if you want to pass an instantiated algorithm configuration like

.. code-block:: python

        alg_config = A2CAlgorithmConfig(
            n_epochs=1,
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

then

.. tabs::

    .. code-tab:: python do this,

        rc = RunContext(algorithm=alg_config)

    .. code-tab:: python not this!

        rc = RunContext(overrides={"algorithm": alg_config})


Further examples of how to use Maze with both the CLI and the high-level API can be found :ref:`here <maze_trainers>`.


**Training**

Training is straightforward with an initialized :class:`~maze.api.run_context.RunContext`:

.. code-block:: python

    rc.train()
    # Or with a specified number of epochs:
    rc.train(n_epochs=10)
    
:meth:`~maze.api.run_context.RunContext.train` passes on all accepted arguments to the instantiated trainer. At the very least the number of epochs to train can be specified, everything else depends on the arguments that the corresponding trainer exposes. See :ref:`here <maze_trainers>` for further information on trainers in Maze. If no arguments are specified, Maze uses the default values included in the loaded configuration.
    
**Rollout**

Rollouts are not supported directly yet, but can be implemented manually:

.. code-block:: python

    env_factory = lambda: GymMazeEnv('CartPole-v0')
    rc = run_context.RunContext(env=lambda: env_factory())
    rc.train()

    # Run trained policy.
    env = env_factory()
    obs = env.reset()
    for i in range(10):
        action = rc.compute_action(obs)
        obs, rewards, dones, info = env.step(action)


**Evaluation**

Evaluations are not supported directly yet, but can be implemented manually:

.. code-block:: python

    rc = RunContext(env=lambda: GymMazeEnv('CartPole-v0'))
    rc.train()

    evaluator = RolloutEvaluator(
        # Environment has to be have statistics logging capabilities for RolloutEvaluator.
        eval_env=LogStatsWrapper.wrap(cartpole_env_factory(), logging_prefix="eval"),
        n_episodes=1,
        model_selection=None
    )
    evaluator.evaluate(rc.policy)