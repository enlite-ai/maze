"""Import critic composers to enable import shortcuts. """
from maze.perception.models.critics.base_state_critic_composer import BaseStateCriticComposer
from maze.perception.models.critics.delta_state_critic_composer import DeltaStateCriticComposer
from maze.perception.models.critics.step_state_critic_composer import StepStateCriticComposer
from maze.perception.models.critics.shared_state_critic_composer import SharedStateCriticComposer

from maze.perception.models.critics.base_state_action_critic_composer import BaseStateActionCriticComposer
from maze.perception.models.critics.step_state_action_critic_composer import StepStateActionCriticComposer
from maze.perception.models.critics.shared_state_action_critics_composer import SharedStateActionCriticComposer

StateCriticComposer = StepStateCriticComposer
StateActionCriticComposer = StepStateActionCriticComposer
"""This is a short hand for all single step environments, such that only one state critic is used.
"""
