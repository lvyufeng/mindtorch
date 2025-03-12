import easy_mindspore
from easy_mindspore.primitive import Primitive
from easy_mindspore import Tensor

def test_add_cpu():
    add = Primitive('Add', 'CPU')
    x = Tensor(1)
    y = Tensor(2)
    print(x, y)
    print(add(x, y))
    print(Primitive._instances)

def test_add_ascend():
    add = Primitive('Add', 'Ascend')
    x = Tensor(1)
    y = Tensor(2)
    print(x, y)
    print(add(x, y))
    print(Primitive._instances)

def test_cast_cpu():
    cast = Primitive('Cast', 'CPU')
    x = Tensor(1)
    print(x.dtype)
    print(cast(x, easy_mindspore.float16))
    print(Primitive._instances)

def test_cast_ascend():
    cast = Primitive('Cast', 'Ascend')
    x = Tensor(1)
    print(x.dtype)
    print(cast(x, easy_mindspore.float16))
    print(Primitive._instances)