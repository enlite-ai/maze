.. RL documentation master file, created by
   sphinx-quickstart on Thu May  7 07:14:55 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |github| image:: logos/GitHub_Logo.png
    :class: inline-figure
    :width: 55
    :target: https://github.com/enlite-ai/maze

.. |github_mark| image:: logos/GitHub-Mark-64px.png
    :class: inline-figure
    :width: 20
    :target: https://github.com/enlite-ai/maze/discussions

.. |stackoverflow| image:: logos/stackoverflow.svg
    :class: inline-figure
    :width: 20
    :target: https://stackoverflow.com/questions/tagged/maze-rl

.. |hydra| image:: logos/hydra_logo.png
    :class: inline-figure
    :width: 35
    :target: https://hydra.cc/

.. |kubernetes| image:: logos/Kubernetes_logo.png
    :class: inline-figure
    :width: 20
    :target: https://kubernetes.io/

.. |ray| image:: logos/ray_logo.png
    :class: inline-figure
    :width: 50
    :target: https://ray.io/

.. |gym| image:: logos/gym_logo.png
    :class: inline-figure
    :width: 20
    :target: https://gym.openai.com/

.. |enlite| image:: logos/EnliteAI_noclaim_rgb.svg
    :class: inline-figure
    :width: 100
    :target: https://www.enlite.ai/

.. |email| image:: logos/mail.svg
    :class: inline-figure
    :width: 20
    :target: mailto:office@enlite.ai

Maze: Applied Reinforcement Learning with Python
================================================

.. raw:: html

   <embed>
   <div style="border:2px; border-style:solid; border-color:#afafb6; border-radius: 25px;
               margin: 1.4em; padding: 0.8em; line-height: 1.5; background-color: white;">
      Maze is an application oriented Reinforcement Learning framework with the vision to:
      <ul>
         <li style="margin-top: 10px;">
            Enable AI-based optimization for a wide range of industrial decision processes.</li>
         <li style="margin-top: 5px;">
            Make RL as a technology accessible to industry and developers.</li>
      </ul>
      Our ultimate goal is to cover the complete development life cycle of RL applications ranging from
      simulation engineering up to agent development, training and deployment.
   </div>
   </embed>

.. raw:: html

   <embed>
   <div class="container" style="text-align:center; vertical-align:middle">
      <table style="width:100%">
      <tr>
         <td style="text-align:right; width:50%; vertical-align:middle"><a class="reference external" href="https://github.com/enlite-ai/maze"><img alt="github" class="inline-figure" src="_images/GitHub_Logo.png" style="width: 55px;" /></a></td>
         <td style="text-align:left; width:50%; vertical-align:middle"><a class="github-button" href="https://github.com/enlite-ai/maze" data-size="large" data-show-count="true" aria-label="Star enlite-ai/maze on GitHub">Star</a></td>
      </tr>
      </table>
   </div>
   </embed>

Getting Started
---------------

.. toctree::
   :hidden:
   :maxdepth: 1

   getting_started/installation.rst
   getting_started/first_example.rst
   getting_started/step_by_step_tutorial.rst
   getting_started/api_contents.rst

- For installing Maze just follow the :ref:`installation instructions <installation>`.
- To see Maze in action check out :ref:`a first example <first_example>`
  and our `Getting Started Notebooks <https://github.com/enlite-ai/maze/tree/main/tutorials/notebooks>`_.
- :ref:`Try your own Gym env <tutorial_gym_env>` or visit our :ref:`Maze step-by-step tutorial <env_from_scratch>`.
- Clone this `project template repo <https://github.com/enlite-ai/maze-cartpole>`_ to start your own Maze project.

You can also find an extensive overview of Maze in the :ref:`table of contents <global_table_of_contents>`
as well as the :ref:`API documentation <api_documentation>`.

Spotlights
----------
Below we list of some of Maze's key features.
The list is far from exhaustive but none the less a nice starting point to dive into the framework.

 - Get things rolling by training your environment and rolling out your policy in just a few lines of code with :ref:`Maze' high-level API <run_context>`.
 - Configure your applications and experiments with the :ref:`Hydra config system <hydra>` |hydra|.
 - Design and visualize your policy and value networks with the :ref:`Perception Module <perception_module>`.
 - :ref:`Pre-process <observation_pre_processing>` and :ref:`normalize <observation_normalization>`
   your observations without writing boiler plate code.
 - Stick to your favourite tools and trainers by
   :ref:`combining Maze with other RL frameworks <maze_and_others>`.
 - Although Maze supports more complex :ref:`environment structures <env-hierarchy>` you can of course still
   :ref:`integrate existing Gym environments <tutorial_gym_env>` |gym|.
 - Scale your training runs with Ray |ray| and Kubernetes |kubernetes|.

.. warning::

   This is a preliminary, non-stable release of Maze. It is not yet complete and not all of our interfaces have settled
   yet. Hence, there might be some breaking changes on our way towards the first stable release.

*This project is powered by* |enlite| |
*Any questions or feedback, just get in touch* |email| |github_mark| |stackoverflow|

.. _global_table_of_contents:

Documentation Overview
======================

Below you find an overview of the general Maze framework documentation, which is beyond
the :ref:`API documentation <api_documentation>`.
The listed pages motivate and explain the underlying concepts
but most importantly also provide code snippets and minimum working examples to quickly get you started.

.. toctree::
   :maxdepth: -1
   :caption: Workflow

   workflow/training.rst
   workflow/rollouts.rst
   workflow/rollouts_trajectories_viewer.rst
   workflow/imitation_and_fine_tuning.rst
   workflow/experimenting.rst

.. toctree::
   :maxdepth: -1
   :caption: Policy and Value Networks

   policy_and_value_networks/perception_overview.rst
   policy_and_value_networks/distributions_and_action_heads.rst
   policy_and_value_networks/perception_template_models.rst
   policy_and_value_networks/perception_custom_models.rst

.. toctree::
   :maxdepth: -1
   :caption: Training

   trainers/maze_trainers.rst
   trainers/maze_rllib_runner.rst

.. toctree::
   :maxdepth: -1
   :caption: Concepts and Structure

   concepts_and_structure/policy_and_agent.rst
   concepts_and_structure/env_hierarchy.rst
   concepts_and_structure/event_system.rst
   concepts_and_structure/hydra.rst
   concepts_and_structure/rendering.rst
   concepts_and_structure/struct_envs/overview.rst
   concepts_and_structure/run_context_overview.rst

.. toctree::
   :maxdepth: -1
   :caption: Environment Customization

   environment_customization/customizing_environments.rst
   environment_customization/reward_aggregation.rst
   environment_customization/environment_wrappers.rst
   environment_customization/observation_pre_processing.rst
   environment_customization/observation_normalization.rst

.. toctree::
   :maxdepth: -1
   :caption: Best Practices and Tutorials

   best_practices_and_tutorials/tricks_of_the_trade.rst
   best_practices_and_tutorials/example_cmds.rst
   best_practices_and_tutorials/integrating_gym_environment.rst
   best_practices_and_tutorials/struct_env_tutorial.rst
   best_practices_and_tutorials/maze_and_other_frameworks.rst
   best_practices_and_tutorials/plain_python_training_example_high_level.rst
   best_practices_and_tutorials/plain_python_training_example_low_level.rst

.. toctree::
   :maxdepth: -1
   :caption: Logging and Monitoring

   logging/log_stats_writer_and_tensorboard.rst
   logging/event_kpi_logging.rst
   logging/action_distribution_visualization.rst
   logging/observation_distribution_visualization.rst

.. toctree::
   :maxdepth: -1
   :caption: Scaling the Training Process

   scaling_training/runner_concept.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`