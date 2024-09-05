""" MazeRL init """
import os

from maze.utils.bcolors import BColors

__version__ = "0.2.1"

# fixes this issue (https://github.com/pytorch/pytorch/issues/37377) when using conda
if "MKL_THREADING_LAYER" not in os.environ or os.environ['MKL_THREADING_LAYER'] != 'GNU':
    BColors.print_colored(
        "INFO: Setting MKL_THREADING_LAYER=GNU to avoid PyTorch issues with conda!",
        color=BColors.OKBLUE)
    os.environ['MKL_THREADING_LAYER'] = 'GNU'


def set_num_threads_to(variable: str, threads: int) -> None:
    """set number of threads to 1 to avoid performance drop with distributed environments.

    :param variable: The environment variable name to modify.
    :param threads: The number of threads to set the environment variable to.
    """
    if variable not in os.environ:
        BColors.print_colored(
            f"INFO: Setting {variable}={threads} to avoid performance drop when using distributed environments!",
            color=BColors.OKBLUE)
        os.environ[variable] = f"{threads}"


def limit_library_cpu_usage_to(threads: int) -> None:
    """Set all environment variables to given number of threads.

    :param threads: Number of threads to set.
    """
    set_num_threads_to("CPU_NUM_THREADS", threads)
    set_num_threads_to("OMP_NUM_THREADS", threads)
    set_num_threads_to("OPENBLAS_NUM_THREADS", threads)
    set_num_threads_to("OPENMP_NUM_THREADS", threads)
    set_num_threads_to("MKL_NUM_THREADS", threads)
    set_num_threads_to("VECLIB_MAXIMUM_THREADS", threads)
    set_num_threads_to("NUMEXPR_NUM_THREADS", threads)


# Set the threads to 1
limit_library_cpu_usage_to(1)

