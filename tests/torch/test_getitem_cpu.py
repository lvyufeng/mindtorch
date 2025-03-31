import torch

def test_linspace():
    x = torch.linspace(2, 4, 8)
    for i in x:
        print(i.item())