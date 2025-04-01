.. |hydra_framework| raw:: html

   <a href="https://hydra.cc/" target="_blank">Hydra configuration framework</a>

.. _hydra:

Configuration with Hydra
========================

Here, we explain the configuration scheme of the Maze framework, which
explain how to configure your environment and other
components using YAML files, run your experiments via CLI, and customize
the runs via CLI overrides.

The Maze framework utilizes the |hydra_framework|. These pages aim to give
you a quick overview of how Maze uses Hydra and what its capabilities are, so that
you can get up to speed quickly without prior Hydra knowledge:

.. toctree::
   :maxdepth: 1

   hydra/overview.rst
   hydra/custom_config.rst
   hydra/advanced.rst

:ref:`Hydra: Overview<hydra-overview>` explains the core concepts of configuration
assembly, overrides and Maze runners controlling the CLI jobs.
:ref:`Hydra: Your Own Configuration Files<hydra-custom>` shows how to get started
with your own configuration in your custom projects.
:ref:`Hydra: Advanced Concepts<hydra-advanced>` explain other components and Hydra features
that power Maze configuration under the hood, such as Maze factory, Hydra interpolations
and specializations.
