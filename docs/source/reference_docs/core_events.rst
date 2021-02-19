.. _event_stats_logging_ref:

Event System, Logging & Statistics
==================================

This page contains the reference documentation for the event and logging system.

.. contents:: Overview
    :depth: 1
    :local:
    :backlinks: top

Event System
------------

These are interfaces, classes and utility functions of the event system:

.. currentmodule:: maze.core.events

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~pubsub.Subscriber
    ~pubsub.Pubsub
    ~event_topic_factory.event_topic_factory
    ~event_service.EventScope
    ~event_service.EventService
    ~event_collection.EventCollection
    ~event_record.EventRecord

Event Logging
-------------

These are the components of the event system:

.. currentmodule:: maze.core.log_events

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~step_event_log.StepEventLog
    ~episode_event_log.EpisodeEventLog
    ~kpi_calculator.KpiCalculator
    ~log_events_writer_registry.LogEventsWriterRegistry
    ~log_events_writer.LogEventsWriter
    ~log_events_writer_tsv.LogEventsWriterTSV
    ~log_events_writer_tsv.EventRow
    ~log_events_utils.SimpleEventLoggingSetup
    ~observation_events.ObservationEvents
    ~action_events.DiscreteActionEvents
    ~action_events.ContinuousActionEvents
    ~log_create_figure_functions.create_categorical_plot
    ~log_create_figure_functions.create_histogram
    ~log_create_figure_functions.create_relative_bar_plot
    ~log_create_figure_functions.create_violin_distribution

Statistics Logging
------------------

These are the components of the statistics logging system:

.. currentmodule:: maze.core.log_stats

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~log_stats_env.LogStatsEnv
    ~log_stats_writer_console.LogStatsWriterConsole
    ~log_stats_writer_tensorboard.LogStatsWriterTensorboard

    ~log_stats.LogStatsLevel
    ~log_stats.LogStatsConsumer
    ~log_stats.LogStatsAggregator
    ~log_stats.LogStatsWriter
    ~log_stats.GlobalLogState
    ~log_stats.LogStatsLogger

    ~log_stats.register_log_stats_writer
    ~log_stats.log_stats
    ~log_stats.increment_log_step
    ~log_stats.get_stats_logger

    ~event_decorators.define_step_stats
    ~event_decorators.define_episode_stats
    ~event_decorators.define_epoch_stats
    ~event_decorators.define_stats_grouping
    ~event_decorators.define_plot
    ~reducer_functions.histogram
    ~log_stats.LogStatsValue
    ~log_stats.LogStatsGroup
    ~log_stats.LogStatsKey
    ~log_stats.LogStats