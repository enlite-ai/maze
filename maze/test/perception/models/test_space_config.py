""" Contains perception unit tests """
import os

from maze.perception.models.space_config import SpacesConfig
from maze.test.shared_test_utils.helper_functions import build_dummy_structured_env


def test_space_config():
    """ perception unit test """

    # init structured env
    env = build_dummy_structured_env()

    # init space config
    space_config = SpacesConfig(action_spaces_dict=env.action_spaces_dict,
                                observation_spaces_dict=env.observation_spaces_dict,
                                agent_counts_dict=env.agent_counts_dict)

    # dump and reload
    dump_file = "tmp_space_config.pkl"
    space_config.save(dump_file)
    assert os.path.exists(dump_file)
    space_config.load(dump_file)
    os.remove(dump_file)
