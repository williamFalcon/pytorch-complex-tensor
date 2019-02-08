import pytest
import torch
import numpy as np
from pytorch_complex_tensor import ComplexTensor


# ------------------
# GRAD TESTS
# -------------------
def test_grad():
    """
    Grad calculated first with tensorflow

    :return:
    """

    c = ComplexTensor([[1, 3, 5], [7, 9, 11], [2, 4, 6], [8, 10, 12]])
    c.requires_grad = True

    # simulate some ops
    out = c + 4
    out = out.mm(c.t())

    # calc grad
    out = out.sum()
    out.backward()

    # d_out/dc
    g = c.grad.view(-1).data.numpy()

    # solution (as provided by running same ops in tensorflow)
    """
    tf_c2 = tf.constant([[1+2j, 3+4j, 5+6j], [7+8j,9+10j,11+12j]], dtype=tf.complex64)
    
    with tf.GradientTape() as t:
    t.watch(tf_c2)
    tf_out = tf_c2 + 4
    tf_out = tf.matmul(tf_out, tf.transpose(tf_c2, perm=[1,0]))
    
    tf_y = tf.reduce_sum(tf_out)
    dy_dc2 = t.gradient(tf_y, tf_c2)
    
    # solution
    print(dy_dc2)
    """
    #
    sol = np.asarray([24, 32, 40, 24, 32, 40, -20, -28, -36, -20, -28, -36])
    assert np.array_equal(g, sol)


def test_size():
    # test sizing when init with tensor
    c = ComplexTensor(torch.zeros(4, 3))
    size = c.size()
    n, m = size[-2:]
    assert n == 2
    assert m == 3

    # test sizing when init with dim spec
    c = ComplexTensor(12, 8)
    size = c.size()
    n, m = size[-2:]
    assert n == 12
    assert m == 8


def test_shape():
    # test sizing when init with tensor
    c = ComplexTensor(torch.zeros(4, 3))
    size = c.shape
    n, m = size[-2:]
    assert n == 2
    assert m == 3

    # test sizing when init with dim spec
    c = ComplexTensor(12, 8)
    size = c.shape
    n, m = size[-2:]
    assert n == 12
    assert m == 8


# ------------------
# REDUCE FX TESTS
# ------------------
def test_abs():
    c = ComplexTensor(torch.zeros(4, 3)) + 2
    c = (4+3j) * c
    c = c.abs()
    c = c.view(-1).data.numpy()

    # do the same in numpy
    sol = np.zeros((2, 3)).astype(np.complex64) + 2
    sol = (4+3j) * sol
    sol = np.abs(sol)

    sol = sol.flatten()
    sol = list(sol.real)

    assert np.array_equal(c, sol)


# ------------------
# SUM TESTS
# ------------------
def test_real_scalar_sum():

    c = ComplexTensor(torch.zeros(4, 3))
    c = c + 4
    c = c.view(-1).data.numpy()

    # do the same in numpy
    sol = np.zeros((2, 3)).astype(np.complex64)
    sol = sol + 4
    sol = sol.flatten()
    sol = list(sol.real) + list(sol.imag)

    assert np.array_equal(c, sol)


def test_complex_scalar_sum():
    c = ComplexTensor(torch.zeros(4, 3))
    c = c + (4+3j)
    c = c.view(-1).data.numpy()

    # do the same in numpy
    sol = np.zeros((2, 3)).astype(np.complex64)
    sol = sol + (4+3j)
    sol = sol.flatten()
    sol = list(sol.real) + list(sol.imag)

    assert np.array_equal(c, sol)


def test_real_matrix_sum():

    c = ComplexTensor(torch.zeros(4, 3))
    r = torch.ones(2, 3)
    c = c + r
    c = c.view(-1).data.numpy()

    # do the same in numpy
    sol = np.zeros((2, 3)).astype(np.complex64)
    sol_r = np.ones((2, 3))
    sol = sol + sol_r
    sol = sol.flatten()
    sol = list(sol.real) + list(sol.imag)

    assert np.array_equal(c, sol)


def test_complex_matrix_sum():

    c = ComplexTensor(torch.zeros(4, 3))
    cc = c + c
    cc = cc.view(-1).data.numpy()

    # do the same in numpy
    sol = np.zeros((2, 3)).astype(np.complex64)
    sol = sol + sol
    sol = sol.flatten()
    sol = list(sol.real) + list(sol.imag)

    assert np.array_equal(cc, sol)


# ------------------
# MULT TESTS
# ------------------
def test_scalar_mult():
    c = ComplexTensor(torch.zeros(4, 3)) + 1
    c = c * 4
    c = c.view(-1).data.numpy()

    # do the same in numpy
    sol = np.zeros((2, 3)).astype(np.complex64) + 1
    sol = sol * 4
    sol = sol.flatten()
    sol = list(sol.real) + list(sol.imag)

    assert np.array_equal(c, sol)


def test_scalar_rmult():
    c = ComplexTensor(torch.zeros(4, 3)) + 1
    c = 4 * c
    c = c.view(-1).data.numpy()

    # do the same in numpy
    sol = np.zeros((2, 3)).astype(np.complex64) + 1
    sol = 4 * sol
    sol = sol.flatten()
    sol = list(sol.real) + list(sol.imag)

    assert np.array_equal(c, sol)


def test_complex_mult():
    c = ComplexTensor(torch.zeros(4, 3)) + 1
    c = c * (4+3j)
    c = c.view(-1).data.numpy()

    # do the same in numpy
    sol = np.zeros((2, 3)).astype(np.complex64) + 1
    sol = sol * (4+3j)
    sol = sol.flatten()
    sol = list(sol.real) + list(sol.imag)

    assert np.array_equal(c, sol)


def test_complex_rmult():
    c = ComplexTensor(torch.zeros(4, 3)) + 1
    c = (4+3j) * c
    c = c.view(-1).data.numpy()

    # do the same in numpy
    sol = np.zeros((2, 3)).astype(np.complex64) + 1
    sol = (4+3j) * sol
    sol = sol.flatten()
    sol = list(sol.real) + list(sol.imag)

    assert np.array_equal(c, sol)


def test_complex_complex_ele_mult():
    """
    Complex mtx x complex mtx elementwise multiply
    :return:
    """
    c = ComplexTensor(torch.zeros(4, 3)) + 1
    c = c * c
    c = c.view(-1).data.numpy()

    # do the same in numpy
    sol = np.zeros((2, 3)).astype(np.complex64) + 1
    sol = sol * sol
    sol = sol.flatten()
    sol = list(sol.real) + list(sol.imag)

    assert np.array_equal(c, sol)


def test_complex_real_ele_mult():
    """
    Complex mtx x real mtx elementwise multiply
    :return:
    """
    c = ComplexTensor(torch.zeros(4, 3)) + 1
    r = torch.ones(2, 3) * 2 + 3
    cr = c * r
    cr = cr.view(-1).data.numpy()

    # do the same in numpy
    np_c = np.ones((2, 3)).astype(np.complex64)
    np_r = np.ones((2, 3)) * 2 + 3
    np_cr = np_c * np_r

    # compare
    np_cr = np_cr.flatten()
    np_cr = list(np_cr.real) + list(np_cr.imag)

    assert np.array_equal(np_cr, cr)


# ------------------
# MM TESTS
# ------------------
def test_complex_real_mm():
    """
    Complex mtx x real mtx matrix multiply
    :return:
    """
    c = ComplexTensor(torch.zeros(4, 3)) + 1
    r = torch.ones(2, 3) * 2 + 3
    cr = c.mm(r.t())
    cr = cr.view(-1).data.numpy()

    # do the same in numpy
    np_c = np.ones((2, 3)).astype(np.complex64)
    np_r = np.ones((2, 3)) * 2 + 3
    np_cr = np.matmul(np_c, np_r.T)

    # compare
    np_cr = np_cr.flatten()
    np_cr = list(np_cr.real) + list(np_cr.imag)

    assert np.array_equal(np_cr, cr)


def test_complex_complex_mm():
    """
    Complex mtx x complex mtx matrix multiply
    :return:
    """
    c = ComplexTensor(torch.zeros(4, 3)) + 1
    cc = c.mm(c.t())
    cc = cc.view(-1).data.numpy()

    # do the same in numpy
    np_c = np.ones((2, 3)).astype(np.complex64)
    np_cc = np.matmul(np_c, np_c.T)

    # compare
    np_cc = np_cc.flatten()
    np_cc = list(np_cc.real) + list(np_cc.imag)

    assert np.array_equal(np_cc, cc)


def test_get_item():
    # init random complex numpy and ct tensors
    a = np.random.randint(0, 10, (3, 2, 3))
    a = a * (1+5j)
    ct = ComplexTensor(a)

    # match dim 0
    __assert_tensors_equal(ct[0], a[0])
    __assert_tensors_equal(ct[-1], a[-1])

    # match dim 1
    __assert_tensors_equal(ct[:, 0], a[:, 0])
    __assert_tensors_equal(ct[:, -1], a[:, -1])

    # match dim 2
    __assert_tensors_equal(ct[:, :, 0], a[:, :, 0])
    __assert_tensors_equal(ct[:, :, -1], a[:, :, -1])

    # match ranges
    __assert_tensors_equal(ct[0:1, 0, -2:], a[0:1, 0, -2:])
    __assert_tensors_equal(ct[-1:, -1:, -2:], a[-1:, -1:, -2:])
    __assert_tensors_equal(ct[:-1, :-1, :-2], a[:-1, :-1, :-2])


def __assert_tensors_equal(ct_tensor, np_tensor):
    # assert we have complexTensor
    assert type(ct_tensor) is ComplexTensor

    # assert values are same
    np_tensor = np_tensor.flatten()
    np_tensor = list(np_tensor.real) + list(np_tensor.imag)
    ct_tensor = ct_tensor.flatten().data.numpy()
    assert np.array_equal(np_tensor, ct_tensor)


if __name__ == '__main__':
    pytest.main([__file__])
