.. _perception_reference:

Perception Module
=================

This page contains the reference documentation of :ref:`Maze Perception Module <perception_module>`.

.. contents:: Overview
    :depth: 1
    :local:
    :backlinks: top

.. _perception_blocks_reference:

maze.perception.blocks
----------------------

These are basic neural network building blocks and interfaces:


.. currentmodule:: maze.perception.blocks

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~base.PerceptionBlock
    ~shape_normalization.ShapeNormalizationBlock
    ~inference.InferenceBlock
    ~inference.InferenceGraph

**Feed Forward:** these are built-in feed forward building blocks:

.. currentmodule:: maze.perception.blocks.feed_forward

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~dense.DenseBlock
    ~vgg_conv.VGGConvolutionBlock
    ~strided_conv.StridedConvolutionBlock
    ~graph_conv.GraphConvBlock
    ~graph_attention.GraphAttentionBlock
    ~multi_head_attention.MultiHeadAttentionBlock
    ~point_net.PointNetFeatureBlock
    ~graph_nn.GNNBlock

**Recurrent:** these are built-in recurrent building blocks:

.. currentmodule:: maze.perception.blocks.recurrent

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~lstm.LSTMBlock

**General:** these are build-in general purpose building blocks:

.. currentmodule:: maze.perception.blocks.general

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~flatten.FlattenBlock
    ~correlation.CorrelationBlock
    ~concat.ConcatenationBlock
    ~functional.FunctionalBlock
    ~gap.GlobalAveragePoolingBlock
    ~masked_global_pooling.MaskedGlobalPoolingBlock
    ~multi_index_slicing.MultiIndexSlicingBlock
    ~repeat_to_match.RepeatToMatchBlock
    ~self_attention_conv.SelfAttentionConvBlock
    ~self_attention_seq.SelfAttentionSeqBlock
    ~slice.SliceBlock
    ~action_masking.ActionMaskingBlock
    ~torch_model_block.TorchModelBlock

**Joint:** these are build in joint building blocks combining multiple perception blocks:

.. currentmodule:: maze.perception.blocks.joint_blocks

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~flatten_dense.FlattenDenseBlock
    ~vgg_conv_dense.VGGConvolutionDenseBlock
    ~vgg_conv_gap.VGGConvolutionGAPBlock
    ~strided_conv_dense.StridedConvolutionDenseBlock
    ~lstm_last_step.LSTMLastStepBlock

.. _perception_builders_reference:

maze.perception.builders
------------------------

These are template model builders:

.. currentmodule:: maze.perception.builders

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~base.BaseModelBuilder
    ~concat.ConcatModelBuilder

.. _perception_composers_reference:

maze.perception.models
----------------------

These are model composers and components:

.. currentmodule:: maze.perception.models

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~model_composer.BaseModelComposer
    ~template_model_composer.TemplateModelComposer
    ~custom_model_composer.CustomModelComposer
    ~space_config.SpacesConfig

These are **maze.perception.models.policies**

.. currentmodule:: maze.perception.models.policies

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~base_policy_composer.BasePolicyComposer
    ~probabilistic_policy_composer.ProbabilisticPolicyComposer

There are **maze.perception.models.critics**

.. currentmodule:: maze.perception.models.critics

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~critic_composer_interface.CriticComposerInterface
    ~base_state_critic_composer.BaseStateCriticComposer
    ~shared_state_critic_composer.SharedStateCriticComposer
    ~step_state_critic_composer.StepStateCriticComposer
    ~delta_state_critic_composer.DeltaStateCriticComposer
    ~StateCriticComposer
    ~base_state_action_critic_composer.BaseStateActionCriticComposer
    ~shared_state_action_critics_composer.SharedStateActionCriticComposer
    ~step_state_action_critic_composer.StepStateActionCriticComposer
    ~StateActionCriticComposer


These are **maze.perception.models.build_in** models

.. currentmodule:: maze.perception.models.built_in

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~flatten_concat.FlattenConcatBaseNet
    ~flatten_concat.FlattenConcatPolicyNet
    ~flatten_concat.FlattenConcatStateValueNet

maze.perception.perception_utils
--------------------------------

These are some helper functions when working with the perception module:

.. currentmodule:: maze.perception.perception_utils

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    observation_spaces_to_in_shapes
    flatten_spaces
    stack_and_flatten_spaces
    convert_to_torch
    convert_to_numpy

maze.perception.weight_init
---------------------------

These are some helper functions for initializing model weights:

.. currentmodule:: maze.perception.weight_init

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    make_module_init_normc
    compute_sigmoid_bias
