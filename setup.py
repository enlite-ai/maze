import os
import re

from setuptools import setup, find_namespace_packages

ROOT_DIR = os.path.dirname(__file__)


def find_version(*filepath):
    """ Extract version information from filepath """
    with open(os.path.join(ROOT_DIR, *filepath)) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


setup(name="maze-rl",
      version=find_version("maze", "__init__.py"),
      packages=find_namespace_packages(include=['maze', 'maze.*']),
      include_package_data=True,


      # python 3.5: we run into conflicts with hydra 1.0.4
      # python 3.9: no ray distribution available
      # gym[box2d] and atari-py included in ray[rllib] is not compatible with python 3.8
      python_requires=">=3.6",

      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Education',

          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Software Development :: Libraries :: Python Modules',

          # matches "license" above
          'License :: Other/Proprietary License',

          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
      ],

      install_requires=[
          "tensorboard",
          "pyyaml",
          "requests",
          "pillow",

          # pinned hydra-core and the associated omegaconf version to 1.1.0 pre-release
          "hydra-core == 1.1.0.dev4",
          "omegaconf == 2.1.0.dev21",

          "gym[box2d]; python_version < '3.8'",
          "gym; python_version >= '3.8'",
          "pandas",
          "networkx",
          "matplotlib",
          "seaborn",
          "tqdm",
          "imageio",

          # train
          "redis",
          "cloudpickle",

          # testing
          "pytest >= 6.0.0",
          "pytest-timeout",
          "pytest-redis"
      ],
      extras_require={
          "testing": [
              "pytest >= 6.0.0",
              "pytest-redis"
          ]
      },
      entry_points={"console_scripts": ["maze-run=maze.maze_cli:maze_run",
                                        "maze-hyper-opt=maze.maze_hyper_opt:maze_hyper_opt"]})
