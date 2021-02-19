.. _event_system:

Maze Event System
===================

The Maze event system is a convenient way how to monitor notable events
happening in the environment as the agent is interacting with it.
This page explains the motivation behind it, gives an overview of how
it is used in Maze (pointing to other relevant sections), and briefly
explains how it works under the hood.

Motivation
----------

Standard metrics such as reward and step count provide a high-level
overview of how an agent is doing in an environment, but don't provide
more detailed information on the actual behavior.

On the other hand, visualizing or
otherwise inspecting the full environment state gives very detailed information
on the behavior in some particular time frame, but is difficult to compare
and aggregate across episodes or training runs.

In Maze, event system fills the space between -- providing more information
about environment mechanics than just watching the reward, while making it easy to
log, understand, and compare it across episodes and rollouts.

What is an event?
-----------------

As the name suggests, an event is something notable that happens during the agent-environment
interaction loop. For example, when the inventory is full in the example 2D cutting env,
a piece will be discarded and the corresponding event will be fired:

.. literalinclude:: code_snippets/example_event.py
  :language: PYTHON

As can be seen above, events carry a descriptive name, encapsulate the details
(like the dimensions of the discarded piece), and are part of a topic
(like "inventory events").

While there are some general events that apply to all environments (like reward-related
events or KPIs), in general, environments declare their own topics and events as they
see fit.

To understand how to declare and integrate custom events into your environment,
see the :ref:`adding events and KPIs <env_from_scratch-events>` tutorial.


.. _event_system-usage:

How are events used in Maze?
----------------------------

There are three main things events are used for throughout Maze:

1. **Reward aggregation.** Reward aggregators declare which events
   they desire to observe, and then calculate the reward on top of them.
   This makes it possible to keep reward aggregators decoupled from the environment,
   which means they can be configured and changed easily. (Check out
   :ref:`reward aggregation<reward_aggregation>` and the
   :ref:`tutorial<tutorial-reward>` for more information.)
2. **Statistics and KPIs.**
   Event declarations can be annotated using decorators which specify how they
   should be aggregated on different levels (i.e., step, episode, and epoch). The statistics
   system then aggregates the events into statistics during trainings and rollouts,
   and displays these statistics in Tensorboard and console. This makes it
   possible to understand the agent's behavior much better than if only high-level
   statistics such as reward and step count were observed. (For more information, see
   how statistics are :ref:`logged<logging>` and :ref:`calculated<event_kpi_log>`.)
3. **Raw event data logging.**
   Events and their details are logged in CSV format, which makes them easy to access
   and analyze later via any custom tools. (While the CSV format should be suitable for
   most data-analysis tools out there, it is also possible to extend the logging functionality
   via custom writers if needed.)

For any other custom needs, it is possible to plug into the event system directly
through the :class:`Pubsub <maze.core.events.pubsub.Pubsub>` or
:class:`EventEnvMixin <maze.core.env.event_env_mixin.EventEnvMixin>` interfaces.

PubSub: Dispatching and Observing Events
----------------------------------------

Each core environment maintains its own :class:`Pubsub <maze.core.events.pubsub.Pubsub>`
message broker (stands for publisher-subscriber). Using the broker, it is possible to
register event topics (created as described in the :ref:`tutorial <env_from_scratch-events>`),
register subscribers (which will then collect the dispatched events),
and dispatch events themselves.

.. literalinclude:: code_snippets/pubsub.py
  :language: PYTHON

Note that the subscriber must implement the
:class:`Subscriber <maze.core.events.pubsub.Subscriber>` interface and declare which
events it want to be notified about. This pattern is used by
:class:`RewardAggregators <maze.core.env.reward.RewardAggregatorInterface>`, and
the :ref:`tutorial on adding reward aggregation<tutorial-reward>`
is also a good place to start for any other custom needs.

EventEnvMixin Interface: Querying Events
----------------------------------------

Core environment also records all events dispatched during the last time step
and makes them accessible using the :class:`EventEnvMixin <maze.core.env.event_env_mixin.EventEnvMixin>` interface.
If you only need to query events dispatched during the last timestep, this option
might be more lightweight than registering with the
:class:`Pubsub <maze.core.events.pubsub.Pubsub>` message broker.

.. literalinclude:: code_snippets/event_env.py
  :language: PYTHON

To see the interface in action, you might want to check out the
:class:`LogStatsWrapper <maze.core.wrappers.log_stats_wrapper.LogStatsWrapper>`, which
uses this interface to query events for :ref:`aggregation <event_kpi_log>`.

Where to Go Next
----------------

After understanding the main concepts of the event system, you might want to:

- See how :ref:`reward aggregation<reward_aggregation>` works and how to
  :ref:`implement it in an environment from scratch<tutorial-reward>`
- Check out the :ref:`statistics logging<logging>` in Tensorboard and console
- Review how the :ref:`events and KPI<event_kpi_log>` aggregation works
