import pytest
import torch
import numpy as np
from pytorch_complex_tensor import ComplexTensor
from pytorch_complex_tensor import torch_complex


def test_stack():
    a = ComplexTensor(torch.zeros(4, 3)) + 1
    b = ComplexTensor(torch.zeros(4, 3)) + 2
    c = ComplexTensor(torch.zeros(4, 3)) + 3

    d = torch_complex.stack([a, b, c], dim=0)
    size = d.size()

    assert size[0] == 3 and size[1] == 2 and size[2] == 3


if __name__ == '__main__':
    pytest.main([__file__])

