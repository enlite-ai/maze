# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'Maze'
copyright = '2021, EnliteAI GmbH'
author = 'enliteAI GmbH'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_tabs.tabs'
]

sphinx_tabs_valid_builders = ['linkcheck']

# build the templated autosummary files
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    'policy_and_value_networks/advanced_concepts.rst',
    'policy_and_value_networks/perception_multi_step_environments.rst',
    'best_practices_and_tutorials/safety_and_reliability.rst',
    'scaling_training/kubernetes_cluster_setup.rst',
    'scaling_training/training_workflow_job_management.rst',
    'scaling_training/trainers_and_distribution_types.rst',
    'scaling_training/experience_database.rst',
    'agent_deployment/agent_packaging.rst',
    'agent_deployment/agent_integration.rst',
]

# Master doc file
master_doc = 'index'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_theme
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_theme.get_html_theme_path()]
html_logo = "logos/main_logo.png"

html_theme_options = {
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': False,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom.css',
]

nitpick_ignore = [
    ('py:class', 'gym.Env'),
    ('py:class', 'gym.spaces.Space'),
    ('py:class', 'gym.spaces.Dict'),
    ('py:class', 'gym.spaces.Box'),
    ('py:class', 'gym.spaces.MultiDiscrete'),
    ('py:class', 'gym.spaces.MultiBinary'),
    ('py:class', 'gym.spaces.Discrete'),
    ('py:class', 'pandas.DataFrame'),
    ('py:class', 'omegaconf.DictConfig'),
    ('py:class', 'multiprocessing.context.BaseContext.Queue'),
    ('py:class', 'numpy.random.mtrand.RandomState'),

    # seems like there is a bug when it comes to this definition, either typing.Generator nor collections.abc.Generator
    # worked
    ('py:class', 'Generator[maze.core.events.event_record.EventRecord, None, None]'),

    # probably this type definition is too complex for sphinx to process
    ('py:class', 'callable'),
    ('py:class', 'Tuple[Union[str, int], ...]'),
    ('py:class', 'Union[str, Callable, None]'),
    ('py:class', 'Union[None, float, numpy.ndarray, Any]'),
    ('py:class', 'Union[None, Dict[Any, Any]]'),
    ('py:class', 'Union[List[Union[None, str, Mapping[str, Any], Any]], Mapping[str, Union[None, str, Mapping[str, Any], Any]]]'),
    ('py:class', 'Union[None, str, Mapping[str, Any], Any]'),
    ('py:class', 'Union[None, str, Mapping[str, Any], Any]'),
    ('py:class', 'Callable[[...], Any]'),
    ('py:class', 'Union[None, Dict[Any, Any]]'),
    ('py:class', 'Union[None, Dict[Any, Any]]'),

    # # torch can't be resolved
    ('py:class', 'torch.dtype'),

    # special maze data types
    ('py:class', 'maze.train.trainers.impala.impala_learner.LearnerOutput'),
    ('py:class', 'maze.train.trainers.imitation.parallel_loaded_im_data_set.ActionTuple'),
    ('py:class', 'maze.train.trainers.imitation.parallel_loaded_im_data_set.ExceptionReport'),
    ('py:class', 'maze.train.trainers.common.distributed.redis_clients.Task'),
    ('py:class', 'AlgorithmConfigType'),

    ('py:class', 'BinaryIO'),

    # type template declarations
    ('py:class', 'T'),
    ('py:class', 'EnvType'),
    ('py:class', 'WrapperType'),
    ('py:class', 'BaseType'),
    ('py:class', 'StepRecordType'),

    # hydra data types
    ('py:class', 'hydra.core.utils.JobReturn'),
    ('py:class', 'hydra.core.config_loader.ConfigLoader'),
    ('py:class', 'hydra.TaskFunction'),
    ('py:class', 'hydra.plugins.launcher.Launcher')
]
nitpicky = True


intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

autodoc_mock_imports = [
    "scipy",
    "numpy",
    "pandas",
    "yaml",
    "cloudpickle",

    "torch",
    "tensorboard",
    "ray",

    "gym",
    "omegaconf",
    "hydra",

    "networkx",
    "matplotlib",
    "seaborn",
    "tqdm",
]
