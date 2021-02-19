import os
import pickle

from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.events.event_record import EventRecord


def test_event_record_is_pickleable():
    original = EventRecord(BaseEnvEvents, BaseEnvEvents.reward, dict(value=5))
    fname = "test"

    with open(fname, "wb") as out_f:
        pickle.dump(original, out_f)

    with open(fname, "rb") as in_f:
        pickled = pickle.load(in_f)

    assert pickled.interface_class == original.interface_class
    assert pickled.interface_method == original.interface_method
    assert pickled.attributes == original.attributes

    os.remove(fname)
