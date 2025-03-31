import torch

def test_tensor_set_data():
    x = torch.randn(10, 20, device='npu')
    buffer = torch.zeros(10000, device='npu')
    x_buffer = buffer[:200].view(10, 20)
    x.data = x_buffer
    x.copy_(torch.ones(10, 20, device='npu'))

    print(x)
    print(x_buffer)
    print(buffer)

    y = x.data
    print(x.data_ptr())
    print(y.data_ptr())

    y.copy_(torch.randn(10, 20, device='npu'))
    print(x)
    print(y)
