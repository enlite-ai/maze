import pytest
from hydra.errors import HydraException

from maze.core.utils.factory import Factory
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.dict import ObservationConversion \
    as DummyObservationConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.dict import DictActionConversion


class CustomDummyObservationConversion(DummyObservationConversion):
    """
    ObservationConversion for DummyEnvironment with attribute.
    """

    def __init__(self, attr: int):
        self.attr = attr


def test_returns_if_arg_already_instantiated():
    obs_conv = CustomDummyObservationConversion(attr=1)
    registry = Factory(base_type=DummyObservationConversion)
    obj = registry.instantiate(config=obs_conv)
    assert obj == obj


def test_raises_exception_on_invalid_type():
    with pytest.raises(AssertionError):
        registry = Factory(base_type=DummyObservationConversion)
        registry.instantiate(config=DictActionConversion())


def test_raises_exception_on_invalid_registry_value():
    with pytest.raises(ImportError):
        registry = Factory(base_type=DummyObservationConversion)
        registry.instantiate(config={"_target_": "wrong_key"})
