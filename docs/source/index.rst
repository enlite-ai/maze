.. maze-rl documentation master file, created by
   sphinx-quickstart on Tue Mar 25 13:47:25 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |getting_started_notebooks_link| raw:: html

   <a href="https://github.com/enlite-ai/maze-examples/tree/main/notebooks" target="_blank">Getting Started Notebooks</a>

.. |project_template_repo_link| raw:: html

   <a href="https://github.com/enlite-ai/maze-cartpole" target="_blank">project template repo</a>

.. |hydra| raw:: html

   <a href="https://hydra.cc" target="_blank"><image class="inline-figure" src="_static/logos/logo-hydra.png" style="width: 35px;" /></a>

.. |kubernetes| raw:: html

   <a href="https://kubernetes.io" target="_blank"><image class="inline-figure" src="_static/logos/logo-kubernetes.png" style="width: 20px;" /></a>

.. |ray| raw:: html

   <a href="https://ray.io" target="_blank"><image class="inline-figure" src="_static/logos/logo-ray.png" style="width: 50px;" /></a>

.. |gym| raw:: html

   <a href="https://gymnasium.farama.org" target="_blank"><image class="inline-figure" src="_static/logos/logo-gymnasium.png" style="width: 20px;" /></a>

.. |enlite| raw:: html

   <a href="https://www.enlite.ai" target="_blank"><image class="inline-figure" src="_static/logos/EnliteAI_noclaim_rgb.svg" style="width: 100px; bottom: 0px;" /></a>

.. |github_mark| raw:: html

   <a href="https://github.com/enlite-ai/maze/discussions/" target="_blank"><image class="inline-figure" src="_static/logos/GitHub-Mark-64px.png" style="width: 20px;" /></a>

.. |stackoverflow| raw:: html

   <a href="https://stackoverflow.com/questions/tagged/maze-rl/" target="_blank"><image class="inline-figure" src="_static/logos/stackoverflow.svg" style="width: 20px;" /></a>

.. |email| image:: _static/logos/mail.svg
    :class: inline-figure
    :width: 20
    :target: mailto:office@enlite.ai

Maze: Applied Reinforcement Learning with Python
================================================

.. raw:: html

   <embed>
      <div style="border:2px; border-style:solid; border-color:#afafb6; border-radius: 25px; margin: 1.3em; padding: 1.3em; line-height: 1.5;">
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
      <div>
         <div class="container" style="text-align:center; vertical-align:middle">
            <a href="https://github.com/enlite-ai/maze/" target="_blank"><img alt="GitHub" class="inline-figure" src="_static/logos/logo-github-light-mode.png" style="width: 100px;" /></a>
         </div>
         <div class="container" style="text-align:center; vertical-align:middle">
            <a class="github-button" href="https://github.com/enlite-ai/maze/subscription" data-icon="octicon-eye" data-show-count="true" data-size="large" aria-label="Watch enlite-ai/maze on GitHub" target="_blank">Watch</a>
            <a class="github-button" href="https://github.com/enlite-ai/maze" data-icon="octicon-star" data-show-count="true" data-size="large" aria-label="Star enlite-ai/maze on GitHub" target="_blank">Star</a>
            <a class="github-button" href="https://github.com/enlite-ai/maze/fork" data-icon="octicon-repo-forked" data-show-count="true" data-size="large" aria-label="Fork enlite-ai/maze on GitHub" target="_blank">Fork</a>
         </div>
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
- To see Maze in action check out :ref:`a first example <first_example>`.
- :ref:`Try your own Gymnasium env <tutorial_gym_env>` or visit our :ref:`Maze step-by-step tutorial <env_from_scratch>`.
- Clone this |project_template_repo_link| to start your own Maze project.

You can also find an extensive overview of Maze in the :ref:`table of contents <global_table_of_contents>`
as well as the :ref:`API documentation <api_documentation>`.

Spotlights
----------
Below we list of some of Maze's key features.
The list is far from exhaustive but none the less a nice starting point to dive into the framework.

 - Configure your applications and experiments with the :ref:`Hydra config system <hydra>` |hydra|.
 - Design and visualize your policy and value networks with the :ref:`Perception Module <perception_module>`.
 - :ref:`Pre-process <observation_pre_processing>` and :ref:`normalize <observation_normalization>`
   your observations without writing boiler plate code.
 - Stick to your favourite tools and trainers by
   :ref:`combining Maze with other RL frameworks <maze_and_others>`.
 - Although Maze supports more complex :ref:`environment structures <env-hierarchy>` you can of course still
   :ref:`integrate existing Gymnasium environments <tutorial_gym_env>` |gym|.
 - Scale your training runs with Ray |ray| and Kubernetes |kubernetes|.

.. warning::

   This is a preliminary, non-stable release of Maze. It is not yet complete and not all of our interfaces have settled
   yet. Hence, there might be some breaking changes on our way towards the first stable release.

*This project is powered by* |enlite|

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
   :class: padding-top-15 padding-bottom-15

   workflow/training.rst
   workflow/rollouts.rst
   workflow/deployment.rst
   workflow/rollouts_trajectories_viewer.rst
   workflow/imitation_and_fine_tuning.rst
   workflow/experimenting.rst

.. toctree::
   :maxdepth: -1
   :caption: Policy and Value Networks
   :class: padding-top-15 padding-bottom-15

   policy_and_value_networks/perception_overview.rst
   policy_and_value_networks/distributions_and_action_heads.rst
   policy_and_value_networks/perception_template_models.rst
   policy_and_value_networks/perception_custom_models.rst

.. toctree::
   :maxdepth: -1
   :caption: Training
   :class: padding-top-15 padding-bottom-15

   trainers/maze_trainers.rst

.. toctree::
   :maxdepth: -1
   :caption: Concepts and Structure
   :class: padding-top-15 padding-bottom-15

   concepts_and_structure/policy_and_agent.rst
   concepts_and_structure/env_hierarchy.rst
   concepts_and_structure/event_system.rst
   concepts_and_structure/hydra.rst
   concepts_and_structure/rendering.rst
   concepts_and_structure/struct_envs/overview.rst

.. toctree::
   :maxdepth: -1
   :caption: Environment Customization
   :class: padding-top-15 padding-bottom-15

   environment_customization/customizing_environments.rst
   environment_customization/reward_aggregation.rst
   environment_customization/environment_wrappers.rst
   environment_customization/observation_pre_processing.rst
   environment_customization/observation_normalization.rst

.. toctree::
   :maxdepth: -1
   :caption: Best Practices and Tutorials
   :class: padding-top-15 padding-bottom-15

   best_practices_and_tutorials/tricks_of_the_trade.rst
   best_practices_and_tutorials/example_cmds.rst
   best_practices_and_tutorials/integrating_gym_environment.rst
   best_practices_and_tutorials/struct_env_tutorial.rst
   best_practices_and_tutorials/maze_and_other_frameworks.rst
   best_practices_and_tutorials/plain_python_training_example_low_level.rst

.. toctree::
   :maxdepth: -1
   :caption: Logging and Monitoring
   :class: padding-top-15 padding-bottom-15

   logging/log_stats_writer_and_tensorboard.rst
   logging/event_kpi_logging.rst
   logging/action_distribution_visualization.rst
   logging/observation_distribution_visualization.rst

.. toctree::
   :maxdepth: -1
   :caption: Scaling the Training Process
   :class: padding-top-15 padding-bottom-15

   scaling_training/runner_concept.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`