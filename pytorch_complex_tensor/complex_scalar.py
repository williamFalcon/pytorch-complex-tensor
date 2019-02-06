"""
Thin wrapper for complex scalar.
Main contribution is to use only real part for backward
"""


class ComplexScalar(object):

    def __init__(self, real, imag):
        self._real = real
        self._imag = imag

    @property
    def real(self):
        return self._real

    @property
    def imag(self):
        return self._imag

    def backward(self):
        self._real.backward()

    def __repr__(self):
        return str(complex(self.real.item(), self.imag.item()))

    def __str__(self):
        return self.__repr__()
