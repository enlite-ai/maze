# Maze examples

This repository contains examples for [Maze](https://github.com/enlite-ai/maze), 
including scripts, notebooks and a custom environment implementation for the 
[Step by Step Tutorial](https://maze-rl.readthedocs.io/en/latest/getting_started/step_by_step_tutorial.html).

## Running these examples with Hydra

When running these examples through command line using Hydra (i.e., using 
`maze-run`), please ensure that this directory is available in your Python path,
e.g. using `export PYTHONPATH="$PYTHONPATH:$PWD/`. 
(Alternatively, you can install it as a package using `pip install -e .`.)

If you copy only parts of this tutorial, you'll need to also tell Hydra where to look
for your config files using the `-cd your_project/conf` flag, or add
a search path plugin (similar to the one in `hydra_plugins` in this repository).

For more information and a step by step tutorial for creating custom
Hydra configurations, please check 
[Maze docs](https://maze-rl.readthedocs.io/en/latest/concepts_and_structure/hydra/custom_config.html).