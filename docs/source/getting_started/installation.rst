.. |pip| raw:: html

   <a href="https://pypi.org/project/maze-rl" target="_blank"><image class="inline-figure" src="../_static/logos/python-pip_logo.png" style="width: 28px;" /></a>

.. |github| raw:: html

   <a href="https://github.com/enlite-ai/maze" target="_blank"><image class="inline-figure" src="../_static/logos/logo-github-light-mode.png" style="width: 55px;" /></a>

.. |conda| raw:: html

   <a href="https://docs.conda.io/projects/conda/en/latest" target="_blank"><image class="inline-figure" src="../_static/logos/conda_logo.png" style="width: 20px;" /></a>

.. |install_pytorch| raw:: html

   <a href="https://pytorch.org/get-started/locally" target="_blank">install PyTorch</a>

.. |install_torch_scatter| raw:: html

   <a href="https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-pip-wheels" target="_blank">the instructions here</a>

.. |install_pygraphviz| raw:: html

   <a href="https://pygraphviz.github.io" target="_blank">this page</a>

.. |open_ai_issue| raw:: html

   <a href="https://github.com/openai/gym/issues/2138" target="_blank">this OpenAI gym issue </a>

.. |building_pillow| raw:: html

   <a href="https://pillow.readthedocs.io/en/stable/installation.html#building-on-linux" target="_blank">building pillow</a>

.. |install_swig| raw:: html

   <a href="https://www.swig.org" target="_blank">swig</a>

.. |install_box2d_py| raw:: html

   <a href="https://pypi.org/project/box2d-py" target="_blank">box2d-py</a>

.. role:: raw-html(raw)
   :format: html

.. _installation:

Installation
============

|pip| To install Maze with pip, run:

.. code-block:: bash

    pip install -U maze-rl

.. note::
    Pip does not |install_pytorch|, you need to make sure it is
    available in your Python environment.

.. note::
    For the graph neural network perception blocks you also need to install **torch_scatter** by following |install_torch_scatter|.

.. note::
    The graphical representation of models requires **pygraphviz** to be installed, see |install_pygraphviz| for
    detailed information on the installation instructions.

.. note::
    Maze is compatible with Python 3.9 to 3.10. We encourage you to start with **Python 3.10**. If you intend to use popular
    environments like Atari or Box2D you might need to install a few additional dependencies because of |open_ai_issue|.

    For Debian systems

    .. code-block:: bash

            apt install libjpeg8-dev zlib1g-dev

    more info on |building_pillow|.

.. note::
    Using Box2D environments such as `LunarLander`, additionally require the installation of |install_swig|
    and |install_box2d_py|. For Debian systems this can be simply done with

    .. code-block:: bash

            apt install swig
            pip install gymnasium[box2d]

:raw-html:`</br>`

|github| To install the bleeding-edge development version from GitHub, first clone the repo.

.. code-block:: bash

    git clone https://github.com/enlite-ai/maze.git
    cd maze

Finally, install the project with pip in development mode and you are ready to start developing.

.. code-block:: bash

    pip install -e .

Alternatively you can install with pip directly from the GitHub repository

.. code-block:: bash

    pip install -e git+https://github.com/enlite-ai/maze.git
