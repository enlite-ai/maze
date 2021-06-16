"""Example of how to train a policy with A2C for gym environments using the built in default networks
for both feed forward and recurrent networks."""
from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.rollout.rollout_generator import RolloutGenerator
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.core.wrappers.observation_stack_wrapper import ObservationStackWrapper
from maze.perception.builders import ConcatModelBuilder
from maze.perception.models.template_model_composer import TemplateModelComposer
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv
from maze.train.trainers.a2c.a2c_algorithm_config import A2CAlgorithmConfig
from maze.train.trainers.a2c.a2c_trainer import A2C
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator
from maze.utils.log_stats_utils import setup_logging


def to_rnn_dict_space_environment(env: str, rnn_steps: int) -> GymMazeEnv:
    """Instantiates a structured gym environment with the option to enable rnn support.

    :param env: The name of the gym environment to instantiate.
    :param rnn_steps: The number of previous time steps to consider for selecting the next action.
    :return: The structured gym environment.
    """

    # instantiate structured environment
    env = GymMazeEnv(env=env)

    # add observation stacking for rnn processing
    if rnn_steps > 1:
        stack_config = [{"observation": "observation",
                         "keep_original": False,
                         "tag": None,
                         "delta": False,
                         "stack_steps": rnn_steps}]
        env = ObservationStackWrapper.wrap(env, stack_config=stack_config)

    return env


def main(n_epochs: int, rnn_steps: int) -> None:
    """Trains the cart pole environment with the multi-step a2c implementation.
    """
    env_name = "CartPole-v0"

    # initialize distributed env
    envs = SequentialVectorEnv([lambda: to_rnn_dict_space_environment(env=env_name, rnn_steps=rnn_steps)
                                for _ in range(4)],
                               logging_prefix="train")

    # initialize the env and enable statistics collection
    eval_env = SequentialVectorEnv([lambda: to_rnn_dict_space_environment(env=env_name, rnn_steps=rnn_steps)
                                    for _ in range(4)],
                                   logging_prefix="eval")

    # map observations to a modality
    obs_modalities_mappings = {"observation": "feature"}

    # define how to process a modality
    modality_config = dict()
    modality_config["feature"] = {"block_type": "maze.perception.blocks.DenseBlock",
                                  "block_params": {"hidden_units": [32, 32],
                                                   "non_lin": "torch.nn.Tanh"}}
    modality_config["hidden"] = {"block_type": "maze.perception.blocks.DenseBlock",
                                 "block_params": {"hidden_units": [64],
                                                  "non_lin": "torch.nn.Tanh"}}
    modality_config["recurrence"] = {}
    if rnn_steps > 0:
        modality_config["recurrence"] = {"block_type": "maze.perception.blocks.LSTMLastStepBlock",
                                         "block_params": {"hidden_size": 8,
                                                          "num_layers": 1,
                                                          "bidirectional": False,
                                                          "non_lin": "torch.nn.Tanh"}}

    template_builder = TemplateModelComposer(
        action_spaces_dict=envs.action_spaces_dict,
        observation_spaces_dict=envs.observation_spaces_dict,
        agent_counts_dict=envs.agent_counts_dict,
        distribution_mapper_config={},
        model_builder=ConcatModelBuilder(modality_config, obs_modalities_mappings, None),
        policy={'_target_': 'maze.perception.models.policies.ProbabilisticPolicyComposer'},
        critic={'_target_': 'maze.perception.models.critics.StateCriticComposer'})

    algorithm_config = A2CAlgorithmConfig(
        n_epochs=n_epochs,
        epoch_length=10,
        patience=10,
        critic_burn_in_epochs=0,
        n_rollout_steps=20,
        lr=0.0005,
        gamma=0.98,
        gae_lambda=1.0,
        policy_loss_coef=1.0,
        value_loss_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=0.0,
        device="cpu",
        rollout_evaluator=RolloutEvaluator(eval_env=eval_env, n_episodes=1,
                                           model_selection=None, deterministic=True)
    )

    model = TorchActorCritic(
        policy=TorchPolicy(networks=template_builder.policy.networks,
                           distribution_mapper=template_builder.distribution_mapper, device=algorithm_config.device),
        critic=template_builder.critic,
        device=algorithm_config.device)

    a2c = A2C(rollout_generator=RolloutGenerator(envs),
              evaluator=algorithm_config.rollout_evaluator,
              algorithm_config=algorithm_config,
              model=model,
              model_selection=None)

    setup_logging(job_config=None)

    # train agent
    a2c.train()

    # final evaluation run
    print("Final Evaluation Run:")
    a2c.evaluate()


if __name__ == "__main__":
    """ main """
    main(n_epochs=1000, rnn_steps=5)
