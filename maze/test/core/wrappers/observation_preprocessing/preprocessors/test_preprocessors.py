""" Unit tests for pre-processors. """
import gym
import numpy as np

from maze.core.wrappers.observation_preprocessing.preprocessors.flatten import FlattenPreProcessor
from maze.core.wrappers.observation_preprocessing.preprocessors.one_hot import OneHotPreProcessor
from maze.core.wrappers.observation_preprocessing.preprocessors.resize_img import ResizeImgPreProcessor
from maze.core.wrappers.observation_preprocessing.preprocessors.rgb2gray import Rgb2GrayPreProcessor
from maze.core.wrappers.observation_preprocessing.preprocessors.transpose import TransposePreProcessor
from maze.core.wrappers.observation_preprocessing.preprocessors.unsqueeze import UnSqueezePreProcessor
from maze.test.shared_test_utils.space_testing_utils import build_dict_obs_space


def test_one_hot_pre_processors():
    """ perception test """

    obs_space = build_dict_obs_space()
    space_dict = obs_space.spaces
    sample = obs_space.sample()

    processor = OneHotPreProcessor(observation_space=space_dict["categorical_feature"])
    assert processor.tag() == "one_hot"

    processed = processor.process(sample["categorical_feature"])
    assert processed.shape == (12,)
    assert np.sum(processed) == 1
    assert processed[sample["categorical_feature"]] == 1

    processor = OneHotPreProcessor(observation_space=space_dict["categorical_feature_series"])

    processed = processor.process(sample["categorical_feature_series"])
    assert processed.shape == (64, 12)
    assert np.allclose(np.sum(processed, axis=1), 1)
    for i in range(processed.shape[0]):
        assert processed[i][sample["categorical_feature_series"][i]] == 1


def test_one_hot_pre_processors_discrete():
    """ perception test """
    obs_space = gym.spaces.Discrete(n=10)
    sample = obs_space.sample()

    processor = OneHotPreProcessor(observation_space=obs_space)

    assert processor.processed_shape() == (10,)

    processed = processor.process(sample)
    assert processed.shape == (10,)
    assert np.sum(processed) == 1
    assert processed[sample] == 1


def test_flatten_pre_processors():
    """ perception test """

    obs_space = build_dict_obs_space()
    space_dict = obs_space.spaces
    sample = obs_space.sample()

    processor = FlattenPreProcessor(observation_space=space_dict["feature_series"], num_flatten_dims=2)
    assert processor.tag() == "flatten"
    processed = processor.process(sample["feature_series"])

    assert processed in processor.processed_space()
    assert processor.processed_shape() == processed.shape
    assert processed.shape == (64 * 24,)
    assert np.all(processed[:24] == sample["feature_series"][0])
    assert np.all(processed[-24:] == sample["feature_series"][-1])


def test_transpose_pre_processors():
    """ perception test """

    obs_space = build_dict_obs_space()
    space_dict = obs_space.spaces
    sample = obs_space.sample()

    processor = TransposePreProcessor(observation_space=space_dict["feature_series"], axes=[1, 0])
    assert processor.tag() == "transpose"
    processed = processor.process(sample["feature_series"])

    assert processed in processor.processed_space()
    assert processor.processed_shape() == processed.shape
    assert processed.shape == (24, 64)
    assert processor.tag() == "transpose"
    assert np.all(processed == np.transpose(sample["feature_series"]))


def test_rgb2gray_pre_processors():
    """ perception test """

    obs_space = build_dict_obs_space()
    space_dict = obs_space.spaces
    sample = obs_space.sample()
    sample["image"][:, :, :] = 1.0

    processor = Rgb2GrayPreProcessor(observation_space=space_dict["image"], rgb_dim=-3)
    assert processor.tag() == "rgb2gray"
    processed = processor.process(sample["image"])

    assert processed in processor.processed_space()
    assert processor.processed_shape() == processed.shape
    assert processed.shape == (32, 32)
    assert processor.tag() == "rgb2gray"
    assert np.allclose(processed, np.sum([0.299, 0.587, 0.114]))


def test_resize_img_pre_processors():
    """ perception test """

    obs_space = build_dict_obs_space()
    space_dict = obs_space.spaces
    sample = obs_space.sample()

    processor = ResizeImgPreProcessor(observation_space=space_dict["image_rgb"], target_size=[16, 16], transpose=False)
    assert processor.tag() == "resize_img"
    processed = processor.process(sample["image_rgb"])

    assert processed in processor.processed_space()
    assert processor.processed_shape() == processed.shape
    assert processed.shape == (16, 16, 3)

    processor = ResizeImgPreProcessor(observation_space=space_dict["image"], target_size=[16, 16], transpose=True)
    processed = processor.process(sample["image"])

    assert processed in processor.processed_space()
    assert processor.processed_shape() == processed.shape
    assert processed.shape == (3, 16, 16)


def test_unsqueeze_pre_processors():
    """ perception test """

    obs_space = build_dict_obs_space()
    space_dict = obs_space.spaces
    sample = obs_space.sample()

    processor = UnSqueezePreProcessor(observation_space=space_dict["image"], dim=-3)
    assert processor.tag() == "unsqueeze"
    processed = processor.process(sample["image"])

    assert processed in processor.processed_space()
    assert processor.processed_shape() == processed.shape
    assert processed.shape == (3, 1, 32, 32)
    assert np.all(processed[:, 0, :, :] == sample["image"])
