import sys
import mindtorch

sys.modules['torch'] = mindtorch
distributed = sys.modules["torch.distributed"] = mindtorch.distributed

__version__ = '2.5.0'