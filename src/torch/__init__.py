import sys
import mindtorch
from mindtorch import *

def load_mod(name):
    exec("import "+name)
    return eval(name)

distributed = sys.modules["torch.distributed"] = mindtorch.distributed
distributions = sys.modules["torch.distributions"] = mindtorch.distributions
library = sys.modules["torch.library"] = load_mod("mindtorch.library")
onnx = sys.modules["torch.onnx"] = load_mod("mindtorch.onnx")
fx = sys.modules["torch.fx"] = load_mod("mindtorch.fx")
jit = sys.modules["torch.jit"] = load_mod("mindtorch.jit")
ao = sys.modules["torch.ao"] = load_mod("mindtorch.ao")
autograd = sys.modules["torch.autograd"] = load_mod("mindtorch.autograd")
_C = sys.modules["torch._C"] = load_mod("mindtorch._C")

sys.modules["torch.utils"] = load_mod("mindtorch.utils")
sys.modules["torch.nn"] = load_mod("mindtorch.nn")
sys.modules["torch.configs"] = load_mod("mindtorch.configs")
sys.modules["torch.ops"] = load_mod("mindtorch.ops")
sys.modules["torch.optim"] = load_mod("mindtorch.optim")
sys.modules["torch.cuda"] = load_mod("mindtorch.cuda")
sys.modules["torch.npu"] = load_mod("mindtorch.npu")
sys.modules["torch.hub"] = load_mod("mindtorch.hub")
sys.modules["torch.types"] = load_mod("mindtorch.types")

sys.modules["torch._utils"] = load_mod("mindtorch._utils")
sys.modules["torch._bind"] = load_mod("mindtorch._bind")
sys.modules["torch._custom_ops"] = load_mod("mindtorch._custom_ops")
sys.modules["torch._dynamo"] = load_mod("mindtorch._dynamo")

__version__ = sys.modules["torch.__version__"] = '2.5.0'