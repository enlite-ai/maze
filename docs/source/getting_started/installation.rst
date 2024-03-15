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
   For the graph neural network perception blocks you also need to install **torch_scatter** by following
   `the instructions here <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-pip-wheels>`_.

.. note::
   The graphical representation of models requires **pygraphviz** to be installed, see `this page <http://pygraphviz.github.io/>`_ for
   detailed information on the installation instructions.

.. note::
    Maze is compatible with Python 3.9 to 3.10. We encourage you to start with **Python 3.10**. If you intend to use popular
    environments like Atari or Box2D you might need to install a few additional dependencies because of `this OpenAI gym issue <https://github.com/openai/gym/issues/2138>`_
    (for Debian systems `sudo apt install libjpeg8-dev zlib1g-dev`, more info on
    `building pillow <https://pillow.readthedocs.io/en/stable/installation.html#building-on-linux>`_).

.. note::
    Using Box2D environments such as `LunarLander`, additionally require the installation of `swig <https://www.swig.org/>`_
    and `box2d-py <https://box2d.org/>`_. This can be done simply via pip with `pip install swig` followed by `pip install box2d-py`.

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
