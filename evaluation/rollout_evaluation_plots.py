"""Render rollout evaluation plots based on given configuration."""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from maze.core.rendering.events_stats_renderer import EventStatsRenderer


def parse_args():
    """Example call (utilising the included example data):

    python rollout_evaluation_plots.py rollout_evaluation_plots_config.yml Example "example_data/event_logs"
    """
    parser = argparse.ArgumentParser("Script for plotting rollout statistics.")
    parser.add_argument('config', help='path to plot config file (args get updated).', type=str, default=None)
    parser.add_argument('experiment_name_and_event_dir', type=str, nargs='+',
                        help='list of experiments to plot passed as '
                             '[[name, path_to_event_log_dir], [name, path_to_event_log_dir], ...].')
    parser.add_argument("--save_figs", action="store_true", help="save generated figures.")
    return parser.parse_args()


def load_data_for_event(event_log_dirs: Dict[str, Path], event_name: str):
    """Load event logs for given events for each experiment and concatenate them into one data frame.

    :param event_log_dirs:  Location of event logs for each experiment. Format of {experiment_name: logs_path}
    :param event_name: Name of event log that we are loading.
    :return: Data frame containing all event logs, with and experiment column specifying the experiment the belong to
    """
    df = None
    for exp_name, exp_event_log_path in event_log_dirs.items():
        exp_df = pd.read_csv(Path(exp_event_log_path) / (event_name + ".tsv"), sep="\t")
        exp_df["experiment"] = exp_name
        df = pd.concat([df, exp_df])
    return df


def plot_timelines(plot_specs: Dict[str, List[Any]], event_log_dirs: Dict[str, Path], subplot_shape: Tuple[int, int]):
    """Render timeline plots (line plots).

    :param plot_specs: Specification of what to plot
    :param event_log_dirs: Where the experiment data are located
    :param subplot_shape: How to layout the subplots
    """
    renderer = EventStatsRenderer()
    for i, (plot_name, spec) in enumerate(plot_specs.items()):
        event_name = spec[0]
        post_processing_func = spec[3]
        if post_processing_func is not None and post_processing_func not in EventStatsRenderer.POST_PROCESSING_FUNCS:
            # Keep the globals here for now to support the custom smoothing
            post_processing_func = globals()[post_processing_func]

        event_log_df = load_data_for_event(event_log_dirs, event_name)

        # Render
        plt.subplot(subplot_shape[0], subplot_shape[1], i + 1)
        renderer.render_timeline_stat(
            df=event_log_df,
            event_name=event_name,
            metric_name=spec[1],
            group_by="experiment",
            aggregation_func=spec[2],
            post_processing_func=post_processing_func
        )

        # Override the default labels
        plt.title("")
        plt.title(plot_name + " ({})".format(spec[2]) if spec[2] is not None else plot_name, fontdict=dict(fontsize=10))
        plt.xlabel("Evaluation day")
        plt.ylabel("")

    plt.tight_layout()
    plt.show()


def plot_kpis(plot_specs: Dict[str, List[Any]], event_log_dirs: Dict[str, Path], subplot_shape: Tuple[int, int]):
    """Render KPI plots.

    :param plot_specs: Specification of what to plot
    :param event_log_dirs: Where the experiment data are located
    :param subplot_shape: How to layout the subplots
    """
    kpis_df = load_data_for_event(event_log_dirs, "BaseEnvEvents.kpi")

    plt.clf()
    for i, (kpi_name, spec) in enumerate(plot_specs.items()):
        plt.subplot(subplot_shape[0], subplot_shape[1], i + 1)
        sns.barplot(x="experiment", y="value", data=kpis_df[kpis_df.name == spec[0]], palette="muted")
        plt.ylabel(kpi_name)
        plt.ylim(spec[1])

    plt.tight_layout()
    plt.show()


def parse_experiments(experiments: List[str]) -> Dict[str, Path]:
    """Parse experiment options provided on command line.

    :param experiments: List of provided options (have even count of items -- odd ones are experiment names,
                        even ones paths to event log dirs).
    :return: Dictionary in the format { experiment_name: event_log_dir_path }
    """
    event_log_dirs = dict()
    assert len(experiments) % 2 == 0, "experiments argument is wrong!"
    n_experiments = len(experiments) // 2
    for i_exp in range(n_experiments):
        name = experiments[2 * i_exp]
        path = Path(experiments[2 * i_exp + 1])
        event_log_dirs[name] = path
    return event_log_dirs


def render_plots(config: Dict[str, Any], event_log_dirs: Dict[str, Path], save_figs: bool = False):
    # Timelines plot
    plt.figure(config["timeline_plot_name"], figsize=config["timeline_plot_figsize"])
    plot_timelines(config["timeline_plot_spec"], event_log_dirs, config["timeline_plot_subplot_shape"])
    if save_figs:
        plt.savefig(config["timeline_plot_name"])

    # KPIs plot
    plt.figure(config["kpi_plot_name"], figsize=config["kpi_plot_figsize"])
    plot_kpis(config["kpi_plot_spec"], event_log_dirs, config["kpi_plot_subplot_shape"])
    if save_figs:
        plt.savefig(config["kpi_plot_name"])


def main():
    """Render timeline and KPI plots according to the provided config."""
    args = parse_args()
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    event_log_dirs = parse_experiments(args.experiment_name_and_event_dir)
    render_plots(config, event_log_dirs, args.save_figs)


if __name__ == "__main__":
    main()
