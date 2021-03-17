"""Provides a function to convert tensorboard logs to a pandas DataFrame.
Implemented as shown in https://github.com/lanpa/tensorboard-dumper/blob/master/dump.py
"""
import struct

import pandas as pd
from tensorboard.compat.proto import event_pb2


def _read(data):
    header = struct.unpack('Q', data[:8])

    event_str = data[12:12 + int(header[0])]  # 8+4
    data = data[12 + int(header[0]) + 4:]
    return data, event_str


def tensorboard_to_pandas(file_path: str) -> pd.DataFrame:
    """Convert the tensorboard log to a pandas DataFrame.

    :param file_path: The path of the tensorboard log file.
    :return: A Pandas DataFrame with the columns "tag", "step", "value" (the fist two are set as index)
    """
    with open(file_path, 'rb') as f:
        data = f.read()

    events = []
    while data:
        data, event_str = _read(data)
        event = event_pb2.Event()

        event.ParseFromString(event_str)
        if event.HasField('summary'):
            for value in event.summary.value:
                if value.HasField('simple_value'):
                    events.append((event.step, value.tag, value.simple_value))

    return pd.DataFrame(events, columns=["step", "tag", "value"]).set_index(["tag", "step"])
