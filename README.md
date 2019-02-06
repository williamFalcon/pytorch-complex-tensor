# pytorch-complex-tensor
torch.Tensor subclass to emulate complex linear algebra.   

Treats first half of tensor as real, second as imaginary.  A few arithmetic operations are implemented to emulate complex arithmetic.   

### Installation
```bash
pip install pytorch-complex-tensor
```

### Example:   
```python   
from pytorch_complex_tensor import ComplexTensor

# equivalent to:
# np.asarray([[1+3j, 1+3j, 1+3j], [2+4j, 2+4j, 2+4j]]).astype(np.complex64)
C = ComplexTensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
C.requires_grad = True

print(C)
# tensor([['(1.0+3.0j)' '(1.0+3.0j)' '(1.0+3.0j)'],
#         ['(2.0+4.0j)' '(2.0+4.0j)' '(2.0+4.0j)']])

# complex absolute value implementation
print(C.abs())
# tensor([[3.1623, 3.1623, 3.1623],
#         [4.4721, 4.4721, 4.4721]], grad_fn=<SqrtBackward>)

# number of complex numbers is half of what it says here
print(C.size())
# torch.Size([4, 3])

# show matrix multiply with real tensor
# also works with complex tensor
x = torch.Tensor([[3, 3], [4, 4], [2, 2]])
xy = C.mm(x)
print(xy)
# tensor([['(9.0+27.0j)' '(9.0+27.0j)'],
#         ['(18.0+36.0j)' '(18.0+36.0j)']])

# show gradients didn't break
xy = xy.sum()

# this is now a complex scalar (thin wrapper with .real, .imag)
print(type(xy))
# pytorch_complex_tensor.complex_scalar.ComplexScalar

print(xy)
# (54+126j)

# calculate dxy / dC
# for complex scalars, grad is wrt the real part
xy.backward()
print(C.grad)
# tensor([['(6.0-0.0j)' '(8.0-0.0j)' '(4.0-0.0j)'],
#         ['(6.0-0.0j)' '(8.0-0.0j)' '(4.0-0.0j)']])
```


### Supported ops:
1. addition 
    - (tensor, scalar). Both complex and real.
2. subtraction 
    - (tensor, scalar). Both complex and real.
3. multiply
    - (tensor, scalar). Both complex and real.
4. mm (matrix multiply)
    - (tensor). Both complex and real.
5. abs (absolute value)
6. t (transpose)