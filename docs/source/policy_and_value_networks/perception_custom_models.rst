.. _custom_models:

Working with Custom Models
==========================
The Maze custom :ref:`model composer <model_composers>` enables us to explicitly specify application specific models
directly in Python. Models can be either written with Maze perception blocks or with plain PyTorch as long as they
inherit from Pytorch’s `nn.Model <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_.

As such models can be easily created, and even existing models from previous work or well known papers can be easily
reused with minor adjustments. However, we recommend to create models using the predefined perception blocks in order
to speed up writing
as well as to take full advantage of features such as shape inference and graphical rendering of the models.

On this page we will first go over the features as well as general working principles. Afterwards we will demonstrate
the custom model composer with three examples:

 - A simple :ref:`feed forward model <custom_example_1>` for cartpole.
 - A more :ref:`complex recurrent network <custom_example_2>` example.
 - The cartpole example again but this time using :ref:`plain PyTorch <custom_example_3>` (that is, no Maze-Perception Blocks).

List of Features
----------------

The custom model composer supports the following features:

 - Specify complex models directly in Python.
 - Supports shape inference and shape checks for a given observation space when relying on Maze perception blocks.
 - Reuse existing PyTorch nn.Models with minor modifications.
 - Stores a graphical rendering of the networks if the inference block is utilized.

.. image:: perception_custom_model_composer.png

.. note::
    All model composers have the single purpose of composing, testing and visualizing the models in code or from a
    config file. After all models have been created and retrieved the model composer will have served its purpose and is
    deleted.

The Custom Models Signature (on Action and Observation Shapes)
--------------------------------------------------------------

As previously mentioned the constraints we impose on any model used in conjunction with the custom model composer are
twofold: Firstly the network class has to inherit from PyTorch's `nn.Model <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_
in order to inherit all network specific
methods and properties such as the *forward* method. Additionally a given network class has to have specified
constructor arguments depending on the type of network.

**Policy Networks** must have the constructor arguments *obs_shapes* and *action_logits_shapes*. When the
models are built in the constructor of the custom model composer these two arguments are passed to the constructor of
the model in addition to any other arbitrary arguments specified. As the name suggests *obs_shapes* is a dictionary
mapping observation names to their corresponding shapes represented as a sequence of integers. Similarly
*action_logits_shapes* is a dictionary that maps action names to their corresponding action
:ref:`distribution logits shapes <action_spaces_and_distributions_module>`. (These shapes are also represented as a
sequence of integers.) Both, observation and action logits shapes are inferred in the model composer utilizing the
*observation_spaces_dict*, *action_spaces_dict* and *distribution_mapper*.

**Critic Networks** require only the constructor argument *obs_shapes*. Any other constructor argument is free for the
user to specify.

To summarize the constraints we impose on custom models:

- Policy Networks:

    - inherit from `nn.Model <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_
    - constructor arguments: *obs_shapes* and *action_logits_shapes*
- Critic Networks:

    - inherit from `nn.Model <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_
    - constructor arguments: *obs_shapes*

.. _custom_example_1:

Example 1: Simple Custom Networks with Perception Blocks
--------------------------------------------------------

Even though designed for more complex models that process multiple observations and predict multiple actions at the
same time you can also compose models for simpler use cases, of course.

In this example we utilize the custom model composer in combination with the perception blocks to compose an
actor-critic model for OpenAI
Gym's `CartPole Env <https://gym.openai.com/envs/CartPole-v0/>`_ using a single dense block in each network.
CartPole has an observation space with dimensionality four and a discrete action space with two options.

The policy model can then be defined as:

.. literalinclude:: code_snippets/custom_cartpole_policy_net.py
  :language: PYTHON

And the critic model as:

.. literalinclude:: code_snippets/custom_cartpole_critic_net.py
  :language: PYTHON

An example config for the model composer could then look like this:

.. literalinclude:: code_snippets/custom_cartpole_net.yaml
  :language: YAML

Details:

 - Models are composed by the :ref:`CustomModelComposer <model_composers>`.
 - No specific action space and probability distribution overrides are specified.
 - Since we are in a single step environment we only have one policy. Additionally we specify the constructor arguments
   we defined in the python code above.
 - For critics we specify the type to be single-step since we are working with a single step environment.
   Furthermore, the network and its constructor arguments are specified.

.. note::
    Although mentioned previously, we want to point out the constructor arguments of the two models again:
    the policy network has the required arguments *obs_shapes* and *action_logits_shapes* in addition to the custom
    arguments *non_lin* and *hidden_units*. The critic network has only the required argument *obs_shapes* and the
    same custom arguments as the policy network.

The resulting inference graphs for a recurrent actor-critic model are shown below:

.. image:: perception_custom_cartpole_policy_network.png
    :width: 49 %
.. image:: perception_custom_cartpole_critic_network.png
    :width: 49 %

.. _custom_example_2:

Example 2: More Complex Custom Networks with Perception Blocks
--------------------------------------------------------------

Now we will consider the more complex example used in the examples of the
:ref:`template model composer <template_feed_forward>`.

The observation space is defined as:

    - *observation_screen* :  a 64 x 64 RGB image
    - *observation_inventory* : a 16-dimensional feature vector

The action space is defined as:

    * *action_move* : a :ref:`categorical action <action_spaces_and_distributions>` with four options deciding to move [*UP, DOWN, LEFT, RIGHT*]
    * *action_use* :  a 16-dimensional :ref:`multi-binary action <action_spaces_and_distributions>` deciding which item to use from inventory

Since we are interested in building a policy and critic network, where both networks should have the same embedding
structure we can create a *base* or *latent space* template:

.. literalinclude:: code_snippets/custom_complex_latent_net.py
  :language: PYTHON

Now using the template we can create the policy:

.. literalinclude:: code_snippets/custom_complex_policy_net.py
  :language: PYTHON

And the critic:

.. literalinclude:: code_snippets/custom_complex_critic_net.py
  :language: PYTHON

An example config for the model composer could then look like this:

.. literalinclude:: code_snippets/custom_complex_net.yaml
  :language: YAML

The resulting inference graphs for a recurrent actor-critic model are shown below. Note that the models are identical
except for the output layers due to the shared base model.

.. image:: perception_custom_complex_policy_network.png
    :width: 49 %
.. image:: perception_custom_complex_critic_network.png
    :width: 49 %

.. _custom_example_3:

Example 3: Custom Networks with (plain PyTorch) Python
------------------------------------------------------

Finally, let's have a look at how we can create a custom model without using any Maze-Perception Components. As already
mentioned, we still have to specify the constructor arguments *obs_shapes* and *action_logits_shapes* but do not need to
use them. Considering again OpenAI Gym's `CartPole Env <https://gym.openai.com/envs/CartPole-v0/>`_ the models could
look like this:

The policy model:

.. literalinclude:: code_snippets/custom_plain_cartpole_policy_net.py
  :language: PYTHON

And the critic model as:

.. literalinclude:: code_snippets/custom_plain_cartpole_critic_net.py
  :language: PYTHON

An example config for the model composer could then look like this:

.. literalinclude:: code_snippets/custom_plain_cartpole_net.yaml
  :language: YAML

.. note::
    Since we do not use the :ref:`inference block <inference_graph_visualization>` in this example, no visual
    representation of the model can be rendered.

Where to Go Next
----------------
 - You can read up on our general introduction to the :ref:`Perception Module <perception_module>`.
 - We explain how to use the :ref:`template model builder <template_models>` in case the you just want to get started
   with training.