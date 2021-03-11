from typing import Any


class RawState:
    """Wrapper class for raw observation -- recorded for envs that do not provide access to actual State object."""

    def __init__(self, observation: Any):
        self.observation = observation


class RawMazeAction:
    """Wrapper class for raw action -- recorded for envs that do not provide access to actual MazeAction object."""

    def __init__(self, action: Any):
        self.action = action