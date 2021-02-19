.. |pip| image:: ../logos/python-pip_logo.png
    :class: inline-figure
    :width: 28

.. |github| image:: ../logos/GitHub_Logo.png
    :class: inline-figure
    :width: 55
    :target: https://github.com/enlite-ai/maze

.. |conda| image:: ../logos/conda_logo.png
    :class: inline-figure
    :width: 20
    :target: https://docs.conda.io/projects/conda/en/latest/index.html

.. |rllib| image:: ../logos/ray_logo.png
    :class: inline-figure
    :width: 55
    :target: https://docs.ray.io/en/master/installation.html

.. _installation:

Installation
============

|pip| To install Maze with pip, run:

.. code:: bash

    pip install maze-rl

.. note::
   Pip does not `install PyTorch <https://pytorch.org/get-started/locally/>`_, you need to make sure it is
   available in your Python environment.

|rllib| If you want to use RLLib it in combination with Maze, optionally install it with

.. code:: bash

    pip install ray[rllib] tensorflow

|github| To install the bleeding-edge development version from github, first clone the repo.

.. code:: bash

    git clone https://github.com/enlite-ai/maze.git
    cd maze

Finally, install the project with pip in development mode and you are good to go and ready to start developing.

.. code:: bash

    pip install -e .

Alternatively you can install with pip directly from the GitHub repository

.. code:: bash

    pip install git+https://github.com/enlite-ai/maze.git
