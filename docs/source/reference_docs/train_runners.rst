.. _train_ref:

Trainers and Training Runners
=============================

This page contains the reference documentation for trainers and training runners:

.. contents:: Overview
    :depth: 2
    :local:
    :backlinks: top

General
-------

These are general interfaces, classes and utility functions for trainers and training runners:

.. currentmodule:: maze.train.trainers

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~common.trainer.Trainer
    ~common.training_runner.TrainingRunner
    ~common.config_classes.TrainConfig
    ~common.config_classes.ModelConfig
    ~common.config_classes.AlgorithmConfig

    ~common.model_selection.model_selection_base.ModelSelectionBase
    ~common.model_selection.best_model_selection.BestModelSelection

    ~common.evaluators.evaluator.Evaluator
    ~common.evaluators.multi_evaluator.MultiEvaluator
    ~common.evaluators.rollout_evaluator.RolloutEvaluator

    ~common.value_transform.ValueTransform
    ~common.value_transform.ReduceScaleValueTransform
    ~common.value_transform.support_to_scalar
    ~common.value_transform.scalar_to_support

    ~common.replay_buffer.replay_buffer.BaseReplayBuffer
    ~common.replay_buffer.uniform_replay_buffer.UniformReplayBuffer

.. _trainers_ref:

Trainers
--------

These are interfaces, classes and utility functions for built-in trainers:

Actor-Critics (AC)
""""""""""""""""""

.. currentmodule:: maze.train.trainers

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~common.actor_critic.actor_critic_runners.ACRunner
    ~common.actor_critic.actor_critic_runners.ACDevRunner
    ~common.actor_critic.actor_critic_runners.ACLocalRunner
    ~common.actor_critic.actor_critic_trainer.ActorCritic
    ~common.actor_critic.actor_critic_events.ActorCriticEvents

    ~a2c.a2c_trainer.A2C
    ~a2c.a2c_algorithm_config.A2CAlgorithmConfig

    ~ppo.ppo_trainer.PPO
    ~ppo.ppo_algorithm_config.PPOAlgorithmConfig

    ~impala.impala_trainer.MultiStepIMPALA
    ~impala.impala_algorithm_config.ImpalaAlgorithmConfig
    ~impala.impala_events.MultiStepIMPALAEvents
    ~impala.impala_runners.ImpalaRunner
    ~impala.impala_runners.ImpalaDevRunner
    ~impala.impala_runners.ImpalaLocalRunner
    ~impala.impala_vtrace.log_probs_from_logits_and_actions_and_spaces
    ~impala.impala_vtrace.from_logits
    ~impala.impala_vtrace.from_importance_weights
    ~impala.impala_vtrace.get_log_rhos

    ~sac.sac_trainer.SACTrainer
    ~sac.sac_algorithm_config.SACAlgorithmConfig
    ~sac.sac_events.SACEvents
    ~sac.sac_runners.SACRunner
    ~sac.sac_runners.SACDevRunner

Evolutionary Strategies (ES)
""""""""""""""""""""""""""""

.. currentmodule:: maze.train.trainers

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~es.es_trainer.ESTrainer
    ~es.es_algorithm_config.ESAlgorithmConfig
    ~es.es_events.ESEvents
    ~es.es_runners.ESMasterRunner
    ~es.es_runners.ESDevRunner
    ~es.es_shared_noise_table.SharedNoiseTable
    ~es.optimizers.base_optimizer.Optimizer
    ~es.optimizers.sgd.SGD
    ~es.optimizers.adam.Adam
    ~es.distributed.es_distributed_rollouts.ESRolloutResult
    ~es.distributed.es_dummy_distributed_rollouts.ESDummyDistributedRollouts
    ~es.distributed.es_distributed_rollouts.ESDistributedRollouts
    ~es.distributed.es_rollout_wrapper.ESAbortException
    ~es.distributed.es_rollout_wrapper.ESRolloutWorkerWrapper
    ~es.es_utils.get_flat_parameters
    ~es.es_utils.set_flat_parameters

Imitation Learning (IL) and Learning from Demonstrations (LfD)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. currentmodule:: maze.train.trainers

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~imitation.imitation_events.ImitationEvents
    ~imitation.bc_runners.BCRunner
    ~imitation.bc_trainer.BCTrainer
    ~imitation.bc_algorithm_config.BCAlgorithmConfig
    ~imitation.bc_validation_evaluator.BCValidationEvaluator
    ~imitation.bc_loss.BCLoss

Utilities
---------

.. currentmodule:: maze.train

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~utils.train_utils.stack_numpy_dict_list
    ~utils.train_utils.unstack_numpy_list_dict
    ~utils.train_utils.compute_gradient_norm
    ~utils.train_utils.stack_torch_dict_list