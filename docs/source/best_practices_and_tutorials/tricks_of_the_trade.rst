.. |email| image:: ../_static/logos/mail.svg
    :class: inline-figure
    :width: 20
    :target: mailto:office@enlite.ai

.. |github_mark| raw:: html

   <a href="https://github.com/enlite-ai/maze/discussions" target="_blank"><image class="inline-figure" src="../_static/logos/GitHub-Mark-64px.png" style="width: 20px;" /></a>

.. |tick| image:: tick.png
    :class: inline-figure
    :width: 18

.. _tips-of-trade:

Tricks of the Trade
===================

This page contains a short list of tips and best practices
that have been quite useful in our work over the last couple of years
and will hopefully also  make it easier for you to train your agents.
However, you should be aware that not each item below will work in each and every application scenario.
Nonetheless, if you are stuck most of them are certainly worth to give a try!

.. note::
    Below you find a subjective and certainly not complete collection of RL tips and tricks
    that will hopefully continue to grow over time.
    However, if you stumble upon something crucial
    that is missing from the list, which you would like to share with the
    RL community and us do not hesitate to get in touch and discuss with us! |email| |github_mark|

Learning and Optimization
-------------------------

|tick| **Action Masking**

Use action masking whenever possible!
This can be crucial as it has the potential to drastically reduce the exploration space of your problem,
which usually leads to a reduced learning time and better overall results.
In some cases action masking also mitigates the need for reward shaping
as invalid actions are excluded from sampling
and there is no need to penalize them with negative rewards any more.
If you want to learn more we recommend to check out the tutorial on
:ref:`structured environments and action masking<struct_env_tutorial>`.

|tick| **Reward Scaling and Shaping**

Make sure that your step rewards are in a reasonable range (e.g., [-1, 1]) not spanning various orders of magnitude.
If these conditions are not fulfilled you might want to apply reward scaling or clipping
(see :class:`RewardScalingWrapper <maze.core.wrappers.reward_scaling_wrapper.RewardScalingWrapper>`,
:class:`RewardClippingWrapper <maze.core.wrappers.reward_clipping_wrapper.RewardClippingWrapper>`)
or :ref:`manually shape your reward <reward_aggregation>`.

|tick| **Reward and Key Performance Indicator (KPI) Monitoring**

When optimizing multi-target objectives (e.g., a weighted sum of sub-rewards)
consider to monitor the contributing rewards on an individual basis.
Even though the overall reward appears to not improve anymore
it might still be the case that the contributing sub-rewards change or fluctuate in the background.
This indicates that the policy and in turn the behaviour of your agent is still changing.
In such settings we recommend to watch the learning progress by :ref:`monitoring KPIs <event_kpi_log>`.


Models and Networks
-------------------

|tick| **Network Design**

Design use case and task specific custom network architectures whenever required.
In a straight forward case this might be a CNN when processing image observations but it could also be
a Graph Convolution Network (GCN) when working with graph or grid observations.
To do so, you might want to check out the :ref:`Perception Module <perception_module>`,
the built-in :ref:`network building blocks <perception_blocks_reference>`
as well as the section on :ref:`how to work with custom models <custom_models>`.

Further, you might want to consider :ref:`behavioural cloning (BC) <maze_trainers-bc>` to design and tweak

- the network architectures
- the observations that are fed into these models

This requires that an imitation learning dataset
fulfilling the pre-conditions for supervised learning is available.
If so, incorporating BC into the model and observation design process can save a lot of time and compute
as you are now training in a supervised learning setting.
*Intuition*: If a network architecture, given the corresponding observations,
is able to fit an offline trajectory dataset (without severe over-fitting)
it might also be a good choice for actual RL training.
If this is relevant to you, you can follow up on how to :ref:`employ imitation learning with Maze <imitation>`.

|tick| **Continuous Action Spaces**

When facing bounded continuous action spaces use
:class:`Squashed Gaussian <maze.distributions.squashed_gaussian.SquashedGaussianProbabilityDistribution>` or
:class:`Beta <maze.distributions.beta.BetaProbabilityDistribution>`
probability distributions for your action heads instead of an unbounded Gaussian.
This avoids action clipping and limits the space of explorable actions to valid regions.
You can learn in the section about
:ref:`distributions and acton heads <action_spaces_and_distributions_module>`
how you can easily switch between different probability distributions using the
:class:`DistributionMapper <maze.distributions.distribution_mapper.DistributionMapper>`.

|tick| **Action Head Biasing**

If you would like to incorporate prior knowledge about the selection frequency of certain actions
you could consider to bias the output layers of these action heads towards the expected sampling distribution
after randomly initializing the weights of your networks
(e.g., :class:`compute_sigmoid_bias <maze.perception.weight_init.compute_sigmoid_bias>`).

Observations
------------

|tick| **Observation Normalization**

For efficient RL training it is crucial that the inputs (e.g. observations) to our models
(e.g. policy and value networks) follow a certain distribution and exhibit values within certain ranges.
To ensure this precondition consider to normalize your observations before actual training by either:

- manually specifying normalization statistics (e.g, divide by 255 for uint8 RGB image observations)
- compute statistics from observations sampled by interacting with the environment

As this is a recurring, boilerplate code heavy task, Maze already provides
:ref:`built-in customizable functionality for normalizing the observations <observation_normalization>`.

|tick| **Observation Pre-Processing**

When feeding categorical observations to your models
consider to convert them to their one-hot encoded vectorized counterparts.
This representation is better suited for neural network processing
and a common practice for example in Natural Language Processing (NLP).
In Maze you can achieve this via :ref:`observation pre-processing <observation_pre_processing>` and the
:class:`OneHotPreProcessor <maze.core.wrappers.observation_preprocessing.preprocessors.one_hot.OneHotPreProcessor>`.
