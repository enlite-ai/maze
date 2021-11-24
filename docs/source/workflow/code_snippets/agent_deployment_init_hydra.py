from maze.core.agent_deployment.agent_deployment import AgentDeployment
from maze.core.utils.config_utils import read_hydra_config

# Note: Needs to be a rollout config (`conf_rollout`), so the policy config is present as well
cfg = read_hydra_config(config_module="maze.conf", config_name="conf_rollout", env="gym_env")

agent_deployment = AgentDeployment(
    policy=cfg.policy,
    env=cfg.env,
    wrappers=cfg.wrappers
)