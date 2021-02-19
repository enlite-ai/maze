"""Simple logging of raw events into TSV files (one file per event type)."""

from pathlib import Path
from typing import Union, Dict, List, Optional

from maze.core.annotations import override
from maze.core.log_events.episode_event_log import EpisodeEventLog
from maze.core.log_events.log_events_writer import LogEventsWriter


class EventRow:
    """Represents one row into the output file for the
    :class:`~maze.core.log_events.log_events_writer_tsv.LogEventsWriterTSV`.

    The purpose of this class is to keep event record attributes together with its episode and step IDs.

    :param episode_id: ID of the episode the event was generated in
    :param env_time: What time the event was generated in (either internal env time, or ID of the step)
    :param attributes: Event attributes dict
    """

    def __init__(self,
                 episode_id: str,
                 env_time: Optional[int],
                 attributes: dict):
        self.episode_id = episode_id
        self.env_time = env_time
        self.attributes = attributes


class LogEventsWriterTSV(LogEventsWriter):
    """
    Writes event logs into TSV files. Each event type has its own file. Each event record
    has associated episode ID and step number.

    :param log_dir: Where event logs should be logged.
    """

    def __init__(self, log_dir: Union[str, Path] = Path("./event_logs")):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @override(LogEventsWriter)
    def write(self, episode_event_log: EpisodeEventLog) -> None:
        """
        Write out provided episode data in to TSV files.
        """

        # Dictionary of events, sorted per event type
        event_tables: Dict[str, List[EventRow]] = {}

        # Split events per type
        for step_event_log in episode_event_log.step_event_logs:
            for event_record in step_event_log.events:
                event_name = event_record.interface_method.__qualname__

                if event_name not in event_tables:
                    event_tables[event_name] = []

                row = EventRow(
                    episode_id=episode_event_log.episode_id,
                    env_time=step_event_log.env_time,
                    attributes=event_record.attributes
                )

                event_tables[event_name].append(row)

        # Write events to TSV files
        for table, rows in event_tables.items():
            file_path = self.log_dir / (table + ".tsv")
            attribute_names = sorted(list(rows[0].attributes.keys()))

            if not file_path.is_file():
                with open(file_path, "w") as out_f:
                    header = ["episode_id", "env_time"] + attribute_names
                    out_f.write("\t".join(header) + '\n')

            with open(file_path, "a") as out_f:
                for row in rows:
                    line = [row.episode_id, row.env_time] + \
                           list(map(lambda attr: row.attributes[attr], attribute_names))
                    line_str = "\t".join(map(lambda x: str(x), line)) + "\n"
                    out_f.write(line_str)
