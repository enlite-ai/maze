""" MazeRL init """
import os

from maze.utils.bcolors import BColors

__version__ = "0.2.0"

# fixes this issue (https://github.com/pytorch/pytorch/issues/37377) when using conda
if "MKL_THREADING_LAYER" not in os.environ or os.environ['MKL_THREADING_LAYER'] != 'GNU':
    BColors.print_colored(
        "INFO: Setting MKL_THREADING_LAYER=GNU to avoid PyTorch issues with conda!",
        color=BColors.OKBLUE)
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

# set number of threads to 1 to avoid performance drop with distributed environments
if "OMP_NUM_THREADS" not in os.environ:
    BColors.print_colored(
        "INFO: Setting OMP_NUM_THREADS=1 to avoid performance drop when using distributed environments!",
        color=BColors.OKBLUE)
    os.environ["OMP_NUM_THREADS"] = "1"

