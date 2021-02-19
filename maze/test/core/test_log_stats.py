from abc import ABC

import numpy as np
import pytest

from maze.core.events.event_record import EventRecord
from maze.core.log_stats.event_decorators import define_step_stats, define_episode_stats, define_epoch_stats, \
    define_stats_grouping
from maze.core.log_stats.log_stats import LogStatsAggregator, LogStatsLevel
from maze.core.log_stats.reducer_functions import histogram


def test_event_attributes():
    """ test the aggregation of individual event attributes """

    class _EventInterface(ABC):
        @define_step_stats(sum, input_name='attr1')
        @define_step_stats(sum, input_name='attr2')
        def event1(self, attr1, attr2):
            pass

    agg = LogStatsAggregator(LogStatsLevel.STEP)
    agg.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(attr1=1, attr2=3)))
    agg.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(attr1=1, attr2=3)))

    stats = agg.reduce()
    assert len(stats) == 2

    value1 = stats[(_EventInterface.event1, "attr1", None)]
    value2 = stats[(_EventInterface.event1, "attr2", None)]

    assert value1 == 2
    assert value2 == 6


def test_event_counting():
    """ test counting as a simple aggregation that operates on the attributes dict """

    class _EventInterface(ABC):
        @define_step_stats(len)
        def event1(self, attr1, attr2):
            pass

    agg = LogStatsAggregator(LogStatsLevel.STEP)
    agg.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(attr1=1, attr2=2)))
    agg.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(attr1=1, attr2=2)))

    stats = agg.reduce()
    assert len(stats) == 1

    key, value = list(stats.items())[0]
    assert value == 2
    # tuple (event, output name)
    assert key == (_EventInterface.event1, None, None)


def test_event_single_attribute():
    """ test if the aggregation function receives scalars if there is only a single event attribute """

    class _EventInterface(ABC):
        @define_step_stats(sum)
        def event1(self, attr1):
            pass

    agg = LogStatsAggregator(LogStatsLevel.STEP)
    agg.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(attr1=1)))
    agg.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(attr1=2)))

    stats = agg.reduce()
    assert len(stats) == 1

    key, value = next(iter(stats.items()))
    assert value == 3
    # tuple (event, output name)
    assert key == (_EventInterface.event1, None, None)


def test_event_skip_aggregation():
    """ test the once-per-step logging """

    class _EventInterface(ABC):
        @define_step_stats(None)
        def event1(self, attr1):
            pass

    agg = LogStatsAggregator(LogStatsLevel.STEP)
    agg.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(attr1=3)))

    stats = agg.reduce()
    assert len(stats) == 1

    key, value = next(iter(stats.items()))
    assert value == 3
    # tuple (event, output name)
    assert key == (_EventInterface.event1, None, None)

    # check if multiple calls per step are correctly detected
    with pytest.raises(AssertionError):
        agg.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(attr1=3)))
        agg.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(attr1=3)))
        agg.reduce()


def test_aggregation_chain():
    """ test the aggregation chain with a single event attribute """

    class _EventInterface(ABC):
        @define_epoch_stats(sum)
        @define_episode_stats(sum)
        @define_step_stats(sum)
        def event1(self, attr1):
            pass

    agg_episode = LogStatsAggregator(LogStatsLevel.EPOCH)
    agg_step = LogStatsAggregator(LogStatsLevel.EPISODE, agg_episode)
    agg_event = LogStatsAggregator(LogStatsLevel.STEP, agg_step)

    no_steps = 5
    no_episodes = 7
    for episode in range(no_episodes):
        for step in range(no_steps):
            agg_event.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(attr1=2)))
            agg_event.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(attr1=3)))
            agg_event.reduce()

        episode_stats = agg_step.reduce()
        assert len(episode_stats) == 1
        value = episode_stats[(_EventInterface.event1, None, None)]
        assert value == no_steps * 5

    epoch_stats = agg_episode.reduce()
    assert len(epoch_stats) == 1
    value = epoch_stats[(_EventInterface.event1, None, None)]
    assert value == no_episodes * no_steps * 5


def test_aggregation_chain_multi_attribute():
    """ test the aggregation chain with two event attributes """

    class _EventInterface(ABC):
        @define_epoch_stats(sum, input_name="attr1")
        @define_epoch_stats(sum, input_name="attr2")
        @define_episode_stats(sum, input_name="attr1")
        @define_episode_stats(sum, input_name="attr2")
        @define_step_stats(sum, input_name="attr1")
        @define_step_stats(sum, input_name="attr2")
        def event1(self, attr1, attr2):
            pass

    agg_episode = LogStatsAggregator(LogStatsLevel.EPOCH)
    agg_step = LogStatsAggregator(LogStatsLevel.EPISODE, agg_episode)
    agg_event = LogStatsAggregator(LogStatsLevel.STEP, agg_step)

    no_steps = 5
    no_episodes = 7
    for episode in range(no_episodes):
        for step in range(no_steps):
            agg_event.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(attr1=2, attr2=-2)))
            agg_event.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(attr1=3, attr2=-3)))
            agg_event.reduce()

        episode_stats = agg_step.reduce()
        assert len(episode_stats) == 2
        value1 = episode_stats[(_EventInterface.event1, "attr1", None)]
        value2 = episode_stats[(_EventInterface.event1, "attr2", None)]
        assert value1 == no_steps * 5
        assert value2 == -no_steps * 5

    epoch_stats = agg_episode.reduce()
    assert len(epoch_stats) == 2
    value1 = epoch_stats[(_EventInterface.event1, "attr1", None)]
    value2 = epoch_stats[(_EventInterface.event1, "attr2", None)]
    assert value1 == no_episodes * no_steps * 5
    assert value2 == -no_episodes * no_steps * 5


def test_aggregation_chain_fork():
    """ test the aggregation chain with two event attributes and different aggregation operations """

    class _EventInterface(ABC):
        @define_epoch_stats(sum, input_name="attr1_sum")
        @define_epoch_stats(np.mean, input_name="attr2_mean")
        @define_episode_stats(sum, input_name="attr1_sum")
        @define_episode_stats(np.mean, input_name="attr2_mean")
        @define_step_stats(sum, input_name="attr1", output_name="attr1_sum")
        @define_step_stats(np.mean, input_name="attr1", output_name="attr1_mean")
        @define_step_stats(sum, input_name="attr2", output_name="attr2_sum")
        @define_step_stats(np.mean, input_name="attr2", output_name="attr2_mean")
        def event1(self, attr1, attr2):
            pass

    agg_episode = LogStatsAggregator(LogStatsLevel.EPOCH)
    agg_step = LogStatsAggregator(LogStatsLevel.EPISODE, agg_episode)
    agg_event = LogStatsAggregator(LogStatsLevel.STEP, agg_step)

    no_steps = 5
    no_episodes = 7
    for episode in range(no_episodes):
        for step in range(no_steps):
            agg_event.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(attr1=2.0, attr2=-2.0)))
            agg_event.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(attr1=3.0, attr2=-3.0)))

            step_stats = agg_event.reduce()
            assert len(step_stats) == 4
            value1_sum = step_stats[(_EventInterface.event1, "attr1_sum", None)]
            value1_mean = step_stats[(_EventInterface.event1, "attr1_mean", None)]
            value2_sum = step_stats[(_EventInterface.event1, "attr2_sum", None)]
            value2_mean = step_stats[(_EventInterface.event1, "attr2_mean", None)]
            assert value1_sum == 5.0
            assert value1_mean == 2.5
            assert value2_sum == -5.0
            assert value2_mean == -2.5

        episode_stats = agg_step.reduce()
        assert len(episode_stats) == 2
        value1 = episode_stats[(_EventInterface.event1, "attr1_sum", None)]
        value2 = episode_stats[(_EventInterface.event1, "attr2_mean", None)]
        assert value1 == no_steps * 5.0
        assert value2 == -2.5

    epoch_stats = agg_episode.reduce()
    assert len(epoch_stats) == 2
    value1 = epoch_stats[(_EventInterface.event1, "attr1_sum", None)]
    value2 = epoch_stats[(_EventInterface.event1, "attr2_mean", None)]
    assert value1 == no_episodes * no_steps * 5.0
    assert value2 == -2.5


def test_grouping():
    """ test the aggregation of individual event attributes """

    class _EventInterface(ABC):
        @define_stats_grouping("group")
        @define_step_stats(sum)
        def event1(self, group, attr1):
            pass

    agg = LogStatsAggregator(LogStatsLevel.STEP)
    for v in [1, 3]:
        agg.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(group=0, attr1=v)))
        agg.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(group=1, attr1=v * 2)))

    stats = agg.reduce()
    assert len(stats) == 2

    value1 = stats[(_EventInterface.event1, None, (0,))]
    value2 = stats[(_EventInterface.event1, None, (1,))]

    assert value1 == 4
    assert value2 == 8


def test_multi_grouping():
    """ test grouping by three attributes """

    class _EventInterface(ABC):
        @define_stats_grouping("group1", "group2", "group3")
        @define_step_stats(sum)
        def event1(self, group1, group2, group3, attr1):
            pass

    agg = LogStatsAggregator(LogStatsLevel.STEP)
    for i in [1, 8]:
        agg.add_event(EventRecord(_EventInterface, _EventInterface.event1,
                                  dict(group1=1, group2=0, group3=0, attr1=1 * i)))
        agg.add_event(EventRecord(_EventInterface, _EventInterface.event1,
                                  dict(group1=0, group2=1, group3=0, attr1=2 * i)))
        agg.add_event(EventRecord(_EventInterface, _EventInterface.event1,
                                  dict(group1=0, group2=0, group3=1, attr1=4 * i)))

    stats = agg.reduce()
    assert len(stats) == 3

    assert stats[(_EventInterface.event1, None, (1, 0, 0))] == 9
    assert stats[(_EventInterface.event1, None, (0, 1, 0))] == 18
    assert stats[(_EventInterface.event1, None, (0, 0, 1))] == 36


def test_multi_group_projection():
    """ test grouping by three attributes """

    class _EventInterface(ABC):
        @define_stats_grouping("group1", "group2", "group3")
        @define_step_stats(sum, group_by="group1", output_name="g1")
        @define_step_stats(sum, group_by="group2", output_name="g2")
        @define_step_stats(sum, group_by="group3", output_name="g3")
        def event1(self, group1, group2, group3, attr1):
            pass

    agg = LogStatsAggregator(LogStatsLevel.STEP)
    agg.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(group1=1, group2=0, group3=0, attr1=1)))
    agg.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(group1=0, group2=1, group3=0, attr1=2)))
    agg.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(group1=0, group2=0, group3=1, attr1=4)))

    stats = agg.reduce()
    assert len(stats) == 6

    assert stats[(_EventInterface.event1, "g1", (0, None, None))] == 6
    assert stats[(_EventInterface.event1, "g1", (1, None, None))] == 1
    assert stats[(_EventInterface.event1, "g2", (None, 0, None))] == 5
    assert stats[(_EventInterface.event1, "g2", (None, 1, None))] == 2
    assert stats[(_EventInterface.event1, "g3", (None, None, 0))] == 3
    assert stats[(_EventInterface.event1, "g3", (None, None, 1))] == 4


def test_event_stats_histogram_2():
    """ test histogram loggin on an event level """

    class _EventInterface(ABC):
        @define_step_stats(histogram, input_name='attr1')
        @define_step_stats(histogram, input_name='attr2')
        def event1(self, attr1, attr2):
            pass

    agg = LogStatsAggregator(LogStatsLevel.STEP)
    agg.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(attr1=1, attr2=2)))
    agg.add_event(EventRecord(_EventInterface, _EventInterface.event1, dict(attr1=1, attr2=2)))

    stats = agg.reduce()
    assert len(stats) == 2

    value1 = stats[(_EventInterface.event1, "attr1", None)]
    value2 = stats[(_EventInterface.event1, "attr2", None)]

    assert value1 == [1, 1]
    assert value2 == [2, 2]
