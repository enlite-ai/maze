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


def set_num_threads_to(os_instance, variable: str, threads: int):
    """set number of threads to 1 to avoid performance drop with distributed environments"""
    if variable not in os_instance.environ:
        BColors.print_colored(
            f"INFO: Setting {variable}={threads} to avoid performance drop when using distributed environments!",
            color=BColors.OKBLUE)
        os_instance.environ[variable] = f"{threads}"


def limit_library_cpu_usage_to(os_instance, threads: int):
    """Set all environment variables to given number of threads"""
    set_num_threads_to(os_instance, "CPU_NUM_THREADS", threads)
    set_num_threads_to(os_instance, "OMP_NUM_THREADS", threads)
    set_num_threads_to(os_instance, "OPENBLAS_NUM_THREADS", threads)
    set_num_threads_to(os_instance, "OPENMP_NUM_THREADS", threads)
    set_num_threads_to(os_instance, "MKL_NUM_THREADS", threads)
    set_num_threads_to(os_instance, "VECLIB_MAXIMUM_THREADS", threads)
    set_num_threads_to(os_instance, "NUMEXPR_NUM_THREADS", threads)


# Set the threads to 1
limit_library_cpu_usage_to(os, 1)

