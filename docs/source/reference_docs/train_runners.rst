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
    ~common.training_runner.TrainConfig
    ~common.training_runner.ModelConfig
    ~common.training_runner.AlgorithmConfig

    ~common.model_selection.model_selection_base.ModelSelectionBase
    ~common.model_selection.best_model_selection.BestModelSelection

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
    ~common.actor_critic.actor_critic_trainer.MultiStepActorCritic
    ~common.actor_critic.actor_critic_events.MultiStepActorCriticEvents

    ~a2c.a2c_trainer.MultiStepA2C
    ~a2c.a2c_algorithm_config.A2CAlgorithmConfig

    ~ppo.ppo_trainer.MultiStepPPO
    ~ppo.ppo_algorithm_config.PPOAlgorithmConfig

    ~impala.impala_trainer.MultiStepIMPALA
    ~impala.impala_algorithm_config.ImpalaAlgorithmConfig
    ~impala.impala_events.MultiStepIMPALAEvents
    ~impala.impala_learner.ImpalaLearner
    ~impala.impala_runners.ImpalaRunner
    ~impala.impala_runners.ImpalaDevRunner
    ~impala.impala_runners.ImpalaLocalRunner
    ~impala.impala_batching.batch_outputs_time_major
    ~impala.impala_vtrace.log_probs_from_logits_and_actions_and_spaces
    ~impala.impala_vtrace.from_logits
    ~impala.impala_vtrace.from_importance_weights
    ~impala.impala_vtrace.get_log_rhos

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
    ~imitation.imitation_evaluator.ImitationEvaluator
    ~imitation.imitation_runners.ImitationRunner
    ~imitation.parallel_loaded_im_data_set.ParallelLoadedImitationDataset
    ~imitation.parallel_loaded_im_data_set.DataLoadWorker
    ~imitation.in_memory_data_set.InMemoryImitationDataSet
    ~imitation.bc_trainer.BCTrainer
    ~imitation.bc_algorithm_config.BCAlgorithmConfig
    ~imitation.bc_evaluator.BCEvaluator
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