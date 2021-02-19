""" Import pre-processors to enable import shortcuts. """
from maze.core.wrappers.observation_preprocessing.preprocessors.base import PreProcessor
from maze.core.wrappers.observation_preprocessing.preprocessors.flatten import FlattenPreProcessor
from maze.core.wrappers.observation_preprocessing.preprocessors.transpose import TransposePreProcessor
from maze.core.wrappers.observation_preprocessing.preprocessors.rgb2gray import Rgb2GrayPreProcessor
from maze.core.wrappers.observation_preprocessing.preprocessors.one_hot import OneHotPreProcessor
from maze.core.wrappers.observation_preprocessing.preprocessors.unsqueeze import UnSqueezePreProcessor
from maze.core.wrappers.observation_preprocessing.preprocessors.resize_img import ResizeImgPreProcessor

assert issubclass(FlattenPreProcessor, PreProcessor)
assert issubclass(TransposePreProcessor, PreProcessor)
assert issubclass(FlattenPreProcessor, PreProcessor)
assert issubclass(Rgb2GrayPreProcessor, PreProcessor)
assert issubclass(OneHotPreProcessor, PreProcessor)
assert issubclass(UnSqueezePreProcessor, PreProcessor)
assert issubclass(ResizeImgPreProcessor, PreProcessor)
