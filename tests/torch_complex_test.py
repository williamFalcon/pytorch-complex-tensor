import pytest
import torch
import numpy as np
from pytorch_complex_tensor import ComplexTensor
from pytorch_complex_tensor import torch_complex


def __test_torch_op(complex_op, torch_op):
    a = ComplexTensor(torch.zeros(4, 3)) + 1
    b = ComplexTensor(torch.zeros(4, 3)) + 2
    c = ComplexTensor(torch.zeros(4, 3)) + 3

    d = complex_op([a, b, c], dim=0)
    size = list(d.size())

    # double second to last axis bc we always half it when generating tensors
    size[-2] *= 2

    # compare against regular torch implementation
    r_a = torch.zeros(4, 3)
    r_b = torch.zeros(4, 3)
    r_c = torch.zeros(4, 3)
    r_d = torch_op([r_a, r_b, r_c], dim=0)
    t_size = r_d.size()

    for i in range(len(size)):
        assert size[i] == t_size[i]


def test_stack():
    __test_torch_op(torch_complex.stack, torch.stack)


def test_cat():
    __test_torch_op(torch_complex.cat, torch.cat)


if __name__ == '__main__':
    pytest.main([__file__])

