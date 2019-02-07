import torch
from pytorch_complex_tensor import ComplexTensor


def __graph_copy__(real, imag):
    # return tensor copy but maintain graph connections
    # force the result to be a ComplexTensor
    result = torch.cat([real, imag], dim=-2)
    result.__class__ = ComplexTensor
    return result


def stack(items, *args, **kwargs):
    fx = torch.stack
    r = [x.real for x in items]
    r = fx(r, *args, **kwargs)

    i = [x.imag for x in items]
    i = fx(i, *args, **kwargs)

    return __graph_copy__(r, i)
