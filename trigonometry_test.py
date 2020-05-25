from pytorch_complex_tensor import ComplexTensor

C = ComplexTensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
C.requires_grad = True
print(C)
result_sin = C.sin()

result_cos = C.cos()
result_tan = C.tan()

print('Sin:')
print(result_sin)


print('Cos:')
print(result_cos)


print('tan:')
print(result_tan)
