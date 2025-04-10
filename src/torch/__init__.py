import sys
import mindtorch
from mindtorch import *

sys.modules['torch'] = mindtorch
distributed = sys.modules["torch.distributed"] = mindtorch.distributed

__version__ = '2.5.0'