""" Contains tests for the step-skip-wrapper. """
import maze.test.core.wrappers as wrapper_module
from maze.core.wrappers.step_skip_wrapper import StepSkipWrapper
from maze.test.shared_test_utils.config_testing_utils import load_env_config
from maze.test.shared_test_utils.dummy_env.dummy_core_env import DummyCoreEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_maze_env import DummyEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_struct_env import DummyStructuredEnvironment
from maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.dict_discrete import \
    DictDiscreteActionConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.dict import ObservationConversion


def build_dummy_structured_environment() -> DummyStructuredEnvironment:
    """
    Instantiates the DummyStructuredEnvironment.

    :return: Instance of a DummyStructuredEnvironment
    """

    observation_conversion = ObservationConversion()

    maze_env = DummyEnvironment(
        core_env=DummyCoreEnvironment(observation_conversion.space()),
        action_conversion=[DictDiscreteActionConversion()],
        observation_conversion=[observation_conversion]
    )

    return DummyStructuredEnvironment(maze_env=maze_env)


def assertion_routine(env: StepSkipWrapper) -> None:
    """ Checks if skipping went well. """
    # test application of wrapper
    obs = env.reset()

    n_substeps = len(env.action_spaces_dict)
    step_actions = {}
    for idx in range((env.n_steps + 2) * n_substeps):
        if env._record_actions:
            action = env.action_space.sample()
            if env.skip_mode == 'sticky':
                step_actions[env.actor_id()[0]] = action
            elif env.skip_mode == 'noop':
                step_actions[env.actor_id()[0]] = env.action_conversion.noop_action()
        else:
            action = step_actions[env.actor_id()[0]]

        observation_keys = list(obs.keys())
        if env.actor_id()[0] == 0:
            for key in ['observation_0']:
                assert key in observation_keys
                assert obs[key] in env.observation_spaces_dict[0][key]
        else:
            for key in ['observation_1']:
                assert key in observation_keys
                assert obs[key] in env.observation_spaces_dict[1][key]
        obs = env.step(action)[0]

        assert step_actions[max(env.actor_id()[0] - 1, 0)] == env._step_actions[max(env.actor_id()[0] - 1, 0)]


def test_observation_skipping_wrapper_shadow():
    """ Step skipping unit test """

    # instantiate env
    env = build_dummy_structured_environment()

    # wrapper config
    step_skip_config = {
        'skip_mode': 'sticky',
        'n_steps': 2
    }

    env = StepSkipWrapper.wrap(env, n_steps=step_skip_config["n_steps"], skip_mode=step_skip_config["skip_mode"])

    # test application of wrapper
    assertion_routine(env)


def test_observation_skipping_wrapper_noop():
    """ Step skipping unit test """

    # instantiate env
    env = build_dummy_structured_environment()

    # wrapper config
    step_skip_config = {
        'skip_mode': 'noop',
        'n_steps': 2
    }

    env = StepSkipWrapper.wrap(env, n_steps=step_skip_config["n_steps"], skip_mode=step_skip_config["skip_mode"])

    # test application of wrapper
    assertion_routine(env)


def test_observation_stack_init_from_yaml_config():
    """ Pre-processor unit test """

    # load config
    config = load_env_config(wrapper_module, "dummy_step_skip_config_file.yaml")

    # init environment
    env = build_dummy_structured_environment()
    env = StepSkipWrapper(env, **config["StepSkipWrapper"])

    # test application of wrapper
    assertion_routine(env)
