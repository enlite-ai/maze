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

.. role:: raw-html(raw)
   :format: html

.. _installation:

Installation
============

|pip| To install Maze with pip, run:

.. code:: bash

    pip install -U maze-rl

.. note::
   Pip does not `install PyTorch <https://pytorch.org/get-started/locally/>`_, you need to make sure it is
   available in your Python environment.

.. note::
    Maze is compatible with Python 3.7 to 3.9. We encourage you to start with **Python 3.7**, as many popular environments like
    Atari or Box2D can not easily be installed in newer Python environments. If you use a Python 3.9 environment, you might
    need to install a few additional dependencies because of `this OpenAI gym issue <https://github.com/openai/gym/issues/2138>`_
    (for Debian systems `sudo apt install libjpeg8-dev zlib1g-dev`, more info on
    `building pillow <https://pillow.readthedocs.io/en/stable/installation.html#building-on-linux>`_)


|rllib| If you want to use RLLib it in combination with Maze, optionally install it with:

.. code:: bash

    pip install ray[rllib]==1.4.1 tensorflow

(Installing RLlib is only required if you would like to use the :ref:`Maze RLlib Runner <maze_rllib_runner>`)

:raw-html:`</br>`

|github| To install the bleeding-edge development version from github, first clone the repo.

.. code:: bash

    git clone https://github.com/enlite-ai/maze.git
    cd maze

Finally, install the project with pip in development mode and you are ready to start developing.

.. code:: bash

    pip install -e .

Alternatively you can install with pip directly from the GitHub repository

.. code:: bash

    pip install git+https://github.com/enlite-ai/maze.git
