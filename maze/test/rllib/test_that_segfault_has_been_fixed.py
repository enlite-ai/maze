"""test monkey patch"""
import numpy as np
import pytest

numpy_user_warning_text = ('The given NumPy array is not writeable, and PyTorch does not support non-writeable' +
                           ' tensors. This means you can write to the underlying (supposedly non-writeable) NumPy '
                           'array using the tensor.' +
                           ' You may want to copy the array to protect its data or make it writeable before '
                           'converting it to a tensor')


@pytest.mark.rllib
def test_monkey_patch_convert_to_tensor():
    """The the rllib monkey patch for writable np arrays"""
    assert_that_patch_has_been_applied()


def assert_that_patch_has_been_applied():
    """assert that patch has been applied correctly"""
    input_value = np.array(range(20))
    input_value.flags.writeable = False

    from ray.rllib.utils.torch_ops import convert_to_torch_tensor
    with pytest.warns(None) as record:
        _ = convert_to_torch_tensor(input_value)
    if len(record.list) > 0:
        assert all([numpy_user_warning_text not in str(warning.message) for warning in record.list])
