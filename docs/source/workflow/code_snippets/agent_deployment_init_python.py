from maze.core.agent_deployment.agent_deployment import AgentDeployment
from maze.test.shared_test_utils.dummy_env.agents.dummy_policy import DummyGreedyPolicy
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env

agent_deployment = AgentDeployment(
    policy=DummyGreedyPolicy(),
    env=build_dummy_maze_env()
)
