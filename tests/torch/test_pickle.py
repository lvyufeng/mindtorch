import pickle
import torch

def test_pickle():
    x = torch.randn(3, 3)
    serialized_data = pickle.dumps(x)
    deserialized_obj = pickle.loads(serialized_data)
    assert torch.allclose(x, deserialized_obj)
