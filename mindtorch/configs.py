import os
from packaging import version
import mindspore
from mindspore._c_expression import MSContext # pylint: disable=no-name-in-module, import-error

from .types import device

SOC = MSContext.get_instance().get_ascend_soc_version()
DEVICE_TARGET = mindspore.get_context('device_target')
GENERATOR_SEED = version.parse(mindspore.__version__) >= version.parse('2.3.0')
SUPPORT_ASYNC_DIST_OP = version.parse(mindspore.__version__) >= version.parse('2.4.0')
SUPPORT_VIEW = GENERATOR_SEED
SUPPORT_BF16 = GENERATOR_SEED and '910b' in SOC
ON_ORANGE_PI = '310b' in SOC
USE_PYBOOST = version.parse(mindspore.__version__) >= version.parse('2.3.0') and DEVICE_TARGET == 'Ascend'
DEFAULT_DTYPE = mindspore.float32
DEFAULT_DEVICE = device('cpu')

def set_pyboost(mode: bool):
    """set global pyboost"""
    global USE_PYBOOST
    USE_PYBOOST = mode

def use_pyboost():
    """set global pyboost"""
    return USE_PYBOOST