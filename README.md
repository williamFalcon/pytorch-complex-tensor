# pytorch-complex-tensor
torch.Tensor subclass to emulate complex linear algebra.   

Treats first half of tensor as real, second as imaginary.  A few arithmetic operations are implemented to emulate complex arithmetic.   


### Example:   
```python   

# equivalent to:
# np.asarray([[1+3j, 1+3j, 1+3j], [2+4j, 2+4j, 2+4j]]).astype(np.complex64)
C = ComplexTensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
C.requires_grad = True

# complex version
print(C.abs())

# number of complex numbers is half of what it says here
print(C.size())

# show matrix multiply with real tensor
# also works with complex tensor
x = torch.Tensor([[3, 3], [4, 4], [2, 2]])
xy = C.mm(x)
print(xy)

# show gradients didn't break
xy = xy.sum()
xy.backward()
print(C.grad)
```