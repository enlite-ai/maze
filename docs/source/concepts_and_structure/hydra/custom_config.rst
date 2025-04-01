.. |hydra_config_search_path| raw:: html

   <a href="https://hydra.cc/docs/next/advanced/search_path/" target="_blank">Config Search Path</a>

.. |hydra_search_path_plugins| raw:: html

   <a href="https://hydra.cc/docs/next/advanced/plugins#searchpathplugin/" target="_blank">customizing working directory pattern</a>

.. |hydra_built_in_experiment_configuration| raw:: html

   <a href="https://hydra.cc/docs/next/patterns/configuring_experiments/" target="_blank">Hydra's built-in Experiment Configuration</a>

.. |vector_obs| raw:: html

   <a href="https://github.com/enlite-ai/maze/blob/main/maze/conf/wrappers/vector_obs.yaml/" target="_blank">vector_obs</a>

.. _hydra-custom:

Hydra: Your Own Configuration Files
===================================

We encourage you to add custom config or experiment files in your own project.
These will make it easy for you to launch different versions
of your environments and agents with different parameters.

To be able to use custom configuration files, you first
need to :ref:`create your config module and add it to the Hydra search path<hydra-custom-search_path>`.
Then, you can either

- create :ref:`your own config components<hydra-custom-components>`
  (e.g., when you just need to customize the environment config),
- create dedicated :ref:`experiment config files <hydra-custom-experiments>` based on the default (master) config,
- or :ref:`create your own root config file<hydra-custom-root_config>` if you have more custom needs.

.. _hydra-custom-search_path:

Step 1: Custom Config Module in Hydra Search Path
---------------------------------------------------

For this, first, create a module where your config will reside
(let's say ``your_project.conf``) and place an ``__init__.py`` file in there.

Second, to make your project available to Hydra, make sure it is either installed
using :code:`pip install -e .`, or added to your Python path manually, using for example
:code:`export PYTHONPATH="$PYTHONPATH:$PWD/"` when in the project directory.

As a final step, you need to tell Hydra to look for your config files.
This is can be done either by specifying your config directory along with each
:code:`maze-run` command using the :code:`-cd` flag:

.. code:: bash

    maze-run -cd your_project/conf ...

Or, to avoid specifying this with every command, you can add your config module
to the Hydra search path by creating the following Hydra plugin
(substitute ``your_project.conf`` with your actual config module path):

.. literalinclude:: code_snippets/custom_search_path.py
  :language: python


Now, you can add additional root config files as well as individual components into your
config package.

For more information on search path customization, check |hydra_config_search_path| and |hydra_search_path_plugins|
in Hydra docs.

.. _hydra-custom-components:

Step 2a: Custom Config Components
---------------------------------

If what you are after is only providing custom options for some of the components
Maze configuration uses (e.g., a custom environment configuration), then it suffices to
add these into the relevant directory in your config module and you are good to go.

For example, if you want a custom configuration for the Gym Car Racing env, you might do:

.. literalinclude:: code_snippets/custom_car_racing_config.yaml
  :language: yaml

Then, you might call ``maze-run`` with the ``env=car_racing`` override and it will
load the configuration from your file.

Depending on your needs, you can mix-and-match your custom configurations
with configurations provided by Maze (e.g. use a custom ``env`` configuration
while using a ``wrappers`` or ``models`` configuration provided by Maze).


.. _hydra-custom-experiments:

Step 2b: Experiment Config
--------------------------

Another convenient way to assemble and maintain different configurations of your experiments
is |hydra_built_in_experiment_configuration|.

It allows you to customize experiments by only specifying the changes to the default (master) configuration.
You can for example change the trainer to PPO, the learning rate to 0.0001
and additionally activate the |vector_obs| wrapper stack
by providing the following experiment configuration:

.. literalinclude:: ../../../../maze/conf/experiment/cartpole_ppo_wrappers.yaml
  :language: yaml
  :caption: conf/experiment/cartpole_ppo_wrappers.yaml

To start the experiment from this experiment config file, run:

.. code:: bash

    maze-run -cn conf_train +experiment=cartpole_ppo_wrappers

For more details on experimenting we refer to the :ref:`experiment configuration docs <experimenting>`.


.. _hydra-custom-root_config:

Step 2c: Custom Root Config
---------------------------

If you require even more customization, you will likely need to define your own root config.
This is usually useful for custom projects, as it allows you to create custom
defaults for the individual config groups.

We suggest you start by copying one of the root configs already available in Maze
(like ``conf_rollout`` or ``conf_train``, depending on what you need), and then adding
more required keys or removing those that are not needed. However, it is also
not difficult to start from scratch if you know what you need.

Once you create your root config file (let's say ``your_project/conf/my_own_conf.yaml``),
it suffices to point Hydra to it via the argument ``-cn my_own_conf``, so your
command would look like this (for illustrative purposes):

.. code-block:: bash

    maze-run -cn my_own_conf

Then, all the defaults and available components that Hydra will look for
depend on what you specified in your new root config.

For an overview of root config, check out :ref:`config root & defaults<hydra-overview-config_root>`.


.. _hydra-custom-runners:

Step 3: Custom Runners (Optional)
---------------------------------

If you want to launch different types of jobs than what Maze provides by default, like
implementing a custom training algorithm or deployment scenario that you would like to
run via the CLI, you will benefit from creating a custom :class:`Runner <maze.runner.Runner>`.

You can subclass the desired class in the runner hierarchy (like the
:class:`TrainingRunner <maze.train.trainers.common.training_runner.TrainingRunner>`
if you are implementing a new training scheme, or the general :class:`Runner <maze.runner.Runner>`
for some more general concept). Then, just create a custom config file for the ``runner`` config
group that configures your new class, and you are good to go.


Where to Go Next
----------------

After understanding how custom configuration is done, you might want to:

- Review the :ref:`Hydra overview<hydra-overview>` to see how you should structure your custom configuration
- Read about the :ref:`advanced concepts of Hydra<hydra-advanced>`
