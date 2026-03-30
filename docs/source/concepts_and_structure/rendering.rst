.. _rendering:

Environment Rendering
=====================

In cases when reviewing the statistics and event logs
provided by the :ref:`event system <event_system>` does
not provide enough insight, rendering the environment state
in a particular time step is helpful.

Maze supports two rendering modes:

1. **Rendering online during the rollout.** This is possible simply using
   the sequential rollout runner for a rollout, and setting the rendering
   flag to true using the following overrides: ``runner=sequential runner.render=true``.
2. **Rendering offline, in a Jupyter notebook, from trajectory data
   collected earlier.** For environments which provide a Maze-compatible render,
   :ref:`rollouts <rollouts>` can be rendered and browsed retroactively. Review
   :ref:`collecting and visualizing rollouts <collecting-rollouts>` for more details.
   (Unfortunately, this mode is not yet supported for ordinary Gym envs -- unless
   a custom Maze-compatible renderer is provided.)
