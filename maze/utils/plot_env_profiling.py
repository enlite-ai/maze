"""File holding methods for plotting the env profiling."""
import logging
import os
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)


def read_event_log(input_dir: str, event_name: str) -> Union[pd.DataFrame, None]:
    """Read the event tsv file and return a dataframe.

    :param input_dir: The input directory.
    :param event_name: The event name.

    :return: A pandas dataframe holding the collected events or None if the event tsv file is missing.
    """
    path = os.path.join(input_dir, 'event_logs', event_name)
    if not os.path.exists(path):
        return None

    return pd.read_csv(path, sep='\t')


def print_as_dataframe(total_timings: dict, total_time: float, wrapper_df: pd.DataFrame,
                       maze_env_df: pd.DataFrame, obs_conv_df: pd.DataFrame, act_conv_df: pd.DataFrame,
                       core_env_df: pd.DataFrame) -> None:
    """Print the profiling values as w dataframe.

    :param total_timings: A dictionary holding the cumulative timings of the measured components.
    :param total_time: The overall total time measured of all steps.
    :param wrapper_df: The dataframe holding the wrapper measurements.
    :param maze_env_df: The dataframe holding the maze environment measurements.
    :param obs_conv_df: The dataframe holding the observation conversion measurements.
    :param act_conv_df: The dataframe holding the activation conversion measurements.
    :param core_env_df: The dataframe holding the core environment measurements.
    """
    accumulated_percentages = dict(wrapper_df.groupby('wrapper_name')['per'].mean().sort_index())
    accumulated_percentages.update(
        {'MazeEnv-other': maze_env_df['per'].mean(), 'MazeEnv-ObsConv': obs_conv_df['per'].mean(),
         'MazeEnv-ActConv': act_conv_df['per'].mean(), 'CoreEnv': core_env_df['per'].mean()})
    arr_total_timings = np.array(list(total_timings.values()))
    arr_accumulated_per = np.array(list(accumulated_percentages.values()))
    tt = pd.DataFrame([arr_total_timings, arr_total_timings / total_time, arr_accumulated_per],
                      columns=list(total_timings.keys())).T
    tt.columns = ['Sum of Measured Time [s]', 'Measured Time / Total Time', 'Mean of measured percents']
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print('')
    print(str(tt).replace('\n', '\n\t'))
    print(f'--> Note that the mean of \'measured percentages differs\' to the \'Measured Time / Total Time\' due to '
          f'varying step times.')


def plot_pi_chart(total_time: float, total_steps: int, total_timings: dict, std_timings: dict,
                  title_txt: str, output_file_path: str) -> None:
    """Create a pie chart of the profiling times and save in the experiment directory.

    :param total_time: The total time measured of all steps.
    :param total_steps: The total number of steps.
    :param total_timings: A dictionary holding the cumulative timings of the measured components.
    :param std_timings: A dictionary holding the standard deviations of the measured components.
    :param title_txt: The title of the pie chart.
    :param output_file_path: The output file path.
    """
    for_plotting = {'smaller': 0}
    smaller_keys = []
    for ll, time_spend in total_timings.items():
        per = time_spend / total_time
        if float(per) < 0.02:
            for_plotting['smaller'] += time_spend
            smaller_keys.append(ll)
        else:
            for_plotting[ll] = time_spend

    # Here all profiled parts that take up less than 2% are accumulated together and are plotted as a single pie slice
    # to make the graph readable.
    if for_plotting['smaller'] > 0:
        for_plotting['\n'.join(smaller_keys)] = for_plotting['smaller']
    del for_plotting['smaller']

    labels = []
    for kk in for_plotting.keys():
        if kk in std_timings and for_plotting[kk]/total_time > 0.03:
            labels.append(f'{for_plotting[kk]/total_time * 100:.2f}%\n'
                          f'[$\mu$: {for_plotting[kk] / total_steps:.2f}s, $\sigma$: {std_timings[kk]:.2f}s]')
        else:
            labels.append(f'{for_plotting[kk]/total_time * 100:.2f}%')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    wedges, texts = ax.pie(list(for_plotting.values()), labels=list(for_plotting.keys()), startangle=0,
                                      wedgeprops=dict(width=1))
    # Place the custom labels inside the slices
    for i, wedge in enumerate(wedges):
        # Calculate the angle of the center of the wedge
        angle = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1

        # Calculate the x and y position of the label
        x = np.cos(np.radians(angle)) * 0.6
        y = np.sin(np.radians(angle)) * 0.6

        # Add the label
        ax.text(x, y, labels[i], ha='center', va='center', fontsize=12, color='white')

    plt.title(title_txt)
    plt.tight_layout()
    plt.savefig(output_file_path)
    print(f'\tEnv profiling saved at: {output_file_path}')


def plot_env_profiling(cur_dir: str) -> None:
    """Plot the env profiling events as a pie chart and save it the experiment directory.

    :param cur_dir: The experiment directory.
    """
    # Check if necessary events exists and can be loaded
    full_env_df = read_event_log(cur_dir, 'EnvProfilingEvents.full_env_step_time.tsv')
    core_env_df = read_event_log(cur_dir, 'EnvProfilingEvents.core_env_step_time.tsv')
    maze_env_df = read_event_log(cur_dir, 'EnvProfilingEvents.maze_env_step_time.tsv')
    obs_conv_df = read_event_log(cur_dir, 'EnvProfilingEvents.observation_conv_time.tsv')
    act_conv_df = read_event_log(cur_dir, 'EnvProfilingEvents.action_conv_time.tsv')
    wrapper_df = read_event_log(cur_dir, 'EnvProfilingEvents.wrapper_step_time.tsv')
    if (
        full_env_df is None
        or core_env_df is None
        or maze_env_df is None
        or obs_conv_df is None
        or act_conv_df is None
        or wrapper_df is None
    ):
        logger.debug('Events for environment profiling not recorded')
        return

    print('Running Environment profiling:')

    sub_step_mean = full_env_df['value'].mean()
    print(f'\tAverage Sub-Step time:                {sub_step_mean:.4f}s based on '
          f'{full_env_df["value"].count()} steps')

    flat_step_mean = full_env_df.groupby(['episode_id', 'env_time']).sum()['value']
    print(f'\tAverage Flat-Step time:               {flat_step_mean.mean():.4f}s based on '
          f'{flat_step_mean.count()} steps')

    episode_mean = full_env_df.groupby('episode_id').sum()['value']
    print(f'\tAverage Episode (without reset) time: {episode_mean.mean():.4f}s based on {episode_mean.count()} '
          f'episodes')

    print(f'\tTotal time spend in steps:            {full_env_df["value"].sum():.4f}s based on '
          f'{full_env_df["value"].count()} steps')

    total_time = full_env_df['value'].sum()
    sub_step_count = full_env_df['value'].count()
    total_timings = dict(wrapper_df.groupby('wrapper_name')['time'].sum().sort_index())
    total_timings.update({'MazeEnv-other': maze_env_df['time'].sum(), 'MazeEnv-ObsConv': obs_conv_df['time'].sum(),
                          'MazeEnv-ActConv': act_conv_df['time'].sum(), 'CoreEnv': core_env_df['time'].sum()})
    std_timings = dict(wrapper_df.groupby('wrapper_name')['time'].std().sort_index())
    std_timings.update({'MazeEnv-other': maze_env_df['time'].std(), 'MazeEnv-ObsConv': obs_conv_df['time'].std(),
                        'MazeEnv-ActConv': act_conv_df['time'].std(), 'CoreEnv': core_env_df['time'].std()})

    assert np.isclose(total_time, sum(total_timings.values())), f'{total_time} vs {sum(total_timings.values())}'

    print_as_dataframe(total_timings, total_time, wrapper_df, maze_env_df, obs_conv_df, act_conv_df, core_env_df)

    plot_pi_chart(total_time, sub_step_count, total_timings, std_timings,
                  title_txt=f'Full Sub-step mean time: {sub_step_mean:.4f}s over: {sub_step_count} steps',
                  output_file_path=f'{cur_dir}/env_profiling.png')

    # In case the investigate_time was declared in the core env, the different operations of the core env
    # can be profiled as well. Here not everything has to be specified, thus we calculate the difference to the
    # core env step time and mark it as 'untracked_time'.
    profiling_df = read_event_log(cur_dir, 'EnvProfilingEvents.investigate_time.tsv')
    if profiling_df is not None:
        print(profiling_df.groupby('name')['time'].sum())
        total_timings = dict(profiling_df.groupby('name')['time'].sum())
        std_timings = dict(profiling_df.groupby('name')['time'].std())
        time_core_env = core_env_df['time'].sum()
        untracked_time = time_core_env - sum(total_timings.values())
        assert 'untracked_time' not in total_timings
        total_timings['untracked_time'] = untracked_time
        per_core_env = time_core_env / total_time

        plot_pi_chart(
            time_core_env, sub_step_count, total_timings, std_timings,
            title_txt=(f'CoreEnv (Sub-) step mean time: {time_core_env.sum() / sub_step_count:.4f}s '
                       f'[{per_core_env * 100:.3f}% of flat step] over: {sub_step_count} steps'),
            output_file_path=f'{cur_dir}/core_env_profiling.png')


if __name__ == '__main__':
    plot_env_profiling('/home/anton/maze_runs/homepod/outputs/2024-08-27/13-41-00')
