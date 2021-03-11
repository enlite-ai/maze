""" Contains tests for the pre-processing wrapper. """
import maze.test.core.wrappers.observation_preprocessing as test_preprocessing_module

from maze.core.wrappers.observation_preprocessing.preprocessing_wrapper import PreProcessingWrapper
from maze.test.core.wrappers.observation_preprocessing.preprocessor_testing_utils import \
    build_dummy_structured_environment
from maze.test.shared_test_utils.config_testing_utils import load_env_config


def test_pre_processing_wrapper():
    """ Pre-processor unit test """

    # instantiate env
    env = build_dummy_structured_environment()

    # pre-processor config
    config = {
        "pre_processor_mapping": [
            {"observation": "observation_0_feature_series",
             "_target_": "maze.preprocessors.FlattenPreProcessor",
             "keep_original": True,
             "config": {"num_flatten_dims": 2}},
            {"observation": "observation_1_categorical_feature",
             "_target_": "maze.preprocessors.OneHotPreProcessor",
             "keep_original": False,
             "config": {}}
        ]
    }

    env = PreProcessingWrapper.wrap(env, pre_processor_mapping=config["pre_processor_mapping"])

    # test application of wrapper
    obs = env.reset()
    observation_keys = list(obs.keys())

    assert 'observation_1_categorical_feature' not in observation_keys
    assert 'observation_1_categorical_feature-one_hot' not in observation_keys

    for key in ['observation_0_feature_series',
                'observation_0_feature_series-flatten']:
        assert key in observation_keys
        assert obs[key] in env.observation_spaces_dict[0][key]

    obs = env.step(env.action_space.sample())[0]
    observation_keys = list(obs.keys())

    assert 'observation_0_feature_series' not in observation_keys
    assert 'observation_0_feature_series-flatten' not in observation_keys
    assert 'observation_1_categorical_feature' not in observation_keys

    for key in ['observation_1_categorical_feature-one_hot']:
        assert key in observation_keys
        assert obs[key] in env.observation_spaces_dict[1][key]


def test_preprocessing_init_from_yaml_config():
    """ Pre-processor unit test """

    # load config
    config = load_env_config(test_preprocessing_module, "dummy_config_file.yml")

    # init environment
    env = build_dummy_structured_environment()
    env = PreProcessingWrapper(env, **config["preprocessing_wrapper"])
    assert isinstance(env, PreProcessingWrapper)

    # test application of wrapper
    obs = env.reset()
    observation_keys = list(obs.keys())

    assert 'observation_1_categorical_feature' not in observation_keys
    assert 'observation_1_categorical_feature-one_hot' not in observation_keys

    for key in ['observation_0_feature_series',
                'observation_0_feature_series-dummy']:
        assert key in observation_keys
        assert obs[key] in env.observation_spaces_dict[0][key]

    obs = env.step(env.action_space.sample())[0]
    observation_keys = list(obs.keys())

    assert 'observation_0_feature_series' not in observation_keys
    assert 'observation_0_feature_series-dummy' not in observation_keys
    assert 'observation_1_categorical_feature' not in observation_keys

    for key in ['observation_1_categorical_feature-one_hot']:
        assert key in observation_keys
        assert obs[key] in env.observation_spaces_dict[1][key]


def test_cascaded_preprocessing():
    """ Pre-processor unit test """

    # instantiate env
    env = build_dummy_structured_environment()

    # pre-processor config
    config = {
        "pre_processor_mapping": [
            {"observation": "observation_0_image",
             "_target_": "maze.core.wrappers.observation_preprocessing.preprocessors.rgb2gray.Rgb2GrayPreProcessor",
             "keep_original": False,
             "config": {"rgb_dim": -1}
             },
            {"observation": "observation_0_image-rgb2gray",
             "_target_": "maze.core.wrappers.observation_preprocessing.preprocessors.resize_img.ResizeImgPreProcessor",
             "keep_original": False,
             "config": {"target_size": [16, 16], "transpose": False}
             }
        ]
    }

    env = PreProcessingWrapper.wrap(env, pre_processor_mapping=config["pre_processor_mapping"])

    # test application of wrapper
    obs = env.reset()
    observation_keys = list(obs.keys())

    assert "observation_0_image-rgb2gray" not in observation_keys
    assert "observation_0_image-rgb2gray-resize_img" in observation_keys
