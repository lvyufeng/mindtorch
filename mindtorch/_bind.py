from mindtorch import float16, bfloat16
from .configs import DEFAULT_DTYPE, DEFAULT_DEVICE
from .types import device as device_

AUTO_CAST_DTYE = {
    'cuda': float16,
    'cpu': bfloat16,
    'npu': float16
}

def set_autocast_dtype(device_type, dtype):
    assert device_type in AUTO_CAST_DTYE.keys(), f'{device_type} is not in {AUTO_CAST_DTYE.keys()}'
    AUTO_CAST_DTYE[device_type] = dtype

def get_autocast_dtype(device_type):
    return AUTO_CAST_DTYE[device_type]

def set_default_dtype(dtype):
    """set default dtype"""
    global DEFAULT_DTYPE
    DEFAULT_DTYPE = dtype

def get_default_dtype():
    """get default dtype"""
    return DEFAULT_DTYPE

def set_default_device(device):
    """set default dtype"""
    global DEFAULT_DEVICE
    if isinstance(device, str):
        device = device_(device)
    DEFAULT_DEVICE = device

def get_default_device():
    """get default dtype"""
    return DEFAULT_DEVICE
