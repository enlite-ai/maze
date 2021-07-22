Trajectory Recorder
===================

These are interfaces, classes and utility functions for recording trajectory data:

.. currentmodule:: maze.core.trajectory_recording

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~datasets.in_memory_dataset.InMemoryDataset
    ~datasets.in_memory_dataset.DataLoadWorker

    ~datasets.trajectory_processor.TrajectoryProcessor
    ~datasets.trajectory_processor.IdentityTrajectoryProcessor
    ~datasets.trajectory_processor.DeadEndClippingTrajectoryProcessor

    ~records.spaces_record.SpacesRecord
    ~records.structured_spaces_record.StepKeyType
    ~records.structured_spaces_record.StructuredSpacesRecord
    ~records.state_record.StateRecord
    ~records.trajectory_record.TrajectoryRecord
    ~records.trajectory_record.StateTrajectoryRecord
    ~records.trajectory_record.SpacesTrajectoryRecord

    ~utils.monitoring_setup.MonitoringSetup
    ~utils.trajectory_utils.SimpleTrajectoryRecordingSetup

    ~writers.trajectory_writer_registry.TrajectoryWriterRegistry
    ~writers.trajectory_writer.TrajectoryWriter
    ~writers.trajectory_writer_file.TrajectoryWriterFile
