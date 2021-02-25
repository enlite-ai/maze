""" MazeRL init """
import os
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from maze.utils.bcolors import BColors

__version__ = "0.0.2.dev1"

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


def get_group_option(group: str) -> str:
    """Get the selected option for the given group, as specified in hydra defaults (=yaml file name).

    There seems to be no way to get the defaults out of hydra directly (hydra 1.0-rc2), therefore we
    extract this info from the load history.

    :param group: The group name.
    :return The selected option for the given group.
    """
    for h in GlobalHydra.instance().config_loader().get_load_history():
        if h.filename.startswith(f"{group}/"):
            return h.filename[len(group) + 1:]


# enable substitutions like e.g. ``${group:env}`` (resolves to the yaml file name of the environment)
OmegaConf.register_resolver("group", get_group_option)
