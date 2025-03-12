import easy_mindspore as ems
from easy_mindspore import ops
import time

def test_cast():
    x = ops.randn(10000, 20000)
    y = ops.cast(x, ems.float32, 'Ascend')
    z = ops.cast(y, ems.float32, 'CPU')
    print('del y')
    x = ops.randn(3, 4)
    time.sleep(10)
    y = ops.cast(x, ems.float32, 'Ascend')
    z = ops.cast(y, ems.float32, 'CPU')
