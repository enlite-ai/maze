import pytest

from maze.core.utils.registry import Registry
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
    registry = Registry(base_type=DummyObservationConversion)
    obj = registry.arg_to_obj(arg=obs_conv, config={})
    assert obj == obj


def test_builds_object_from_registry():
    expected = CustomDummyObservationConversion(attr=1)
    registry = Registry(base_type=DummyObservationConversion)
    registry.type_registry = {'CustomDummyObservationConversion': CustomDummyObservationConversion}
    obj = registry.arg_to_obj(arg='CustomDummyObservationConversion', config={'attr': 1})
    assert type(expected) == type(obj)
    assert expected.attr == obj.attr


def test_raises_exception_on_invalid_type():
    with pytest.raises(TypeError):
        registry = Registry(base_type=DummyObservationConversion)
        registry.arg_to_obj(arg=DictActionConversion(), config={})


def test_raises_exception_on_invalid_registry_value():
    with pytest.raises(ValueError):
        registry = Registry(base_type=DummyObservationConversion)
        registry.type_registry = {'CustomDummyObservationConversion': CustomDummyObservationConversion}
        registry.arg_to_obj(arg='wrong_key', config={})
