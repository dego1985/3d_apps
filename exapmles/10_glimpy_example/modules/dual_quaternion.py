from typing import Final, Union

import numpy as np
from modules.ndarray import constant
from modules.ndarray import ndarray as _ndarray


class ndarray(_ndarray):
    dim = 8
    _mul_coefs = {
        (0, 0, 0): 1,
        (1, 1, 0): -1,
        (2, 2, 0): -1,
        (3, 3, 0): -1,
        (0, 1, 1): 1,
        (1, 0, 1): 1,
        (0, 2, 2): 1,
        (2, 0, 2): 1,
        (0, 3, 3): 1,
        (3, 0, 3): 1,
        (1, 2, 3): 1,
        (2, 1, 3): -1,
        (2, 3, 1): 1,
        (3, 2, 1): -1,
        (3, 1, 2): 1,
        (1, 3, 2): -1,
        #
        (0, 0 + 4, 0 + 4): 1,
        (1, 1 + 4, 0 + 4): -1,
        (2, 2 + 4, 0 + 4): -1,
        (3, 3 + 4, 0 + 4): -1,
        (0, 1 + 4, 1 + 4): 1,
        (1, 0 + 4, 1 + 4): 1,
        (0, 2 + 4, 2 + 4): 1,
        (2, 0 + 4, 2 + 4): 1,
        (0, 3 + 4, 3 + 4): 1,
        (3, 0 + 4, 3 + 4): 1,
        (1, 2 + 4, 3 + 4): 1,
        (2, 1 + 4, 3 + 4): -1,
        (2, 3 + 4, 1 + 4): 1,
        (3, 2 + 4, 1 + 4): -1,
        (3, 1 + 4, 2 + 4): 1,
        (1, 3 + 4, 2 + 4): -1,
        #
        (0 + 4, 0, 0 + 4): 1,
        (1 + 4, 1, 0 + 4): -1,
        (2 + 4, 2, 0 + 4): -1,
        (3 + 4, 3, 0 + 4): -1,
        (0 + 4, 1, 1 + 4): 1,
        (1 + 4, 0, 1 + 4): 1,
        (0 + 4, 2, 2 + 4): 1,
        (2 + 4, 0, 2 + 4): 1,
        (0 + 4, 3, 3 + 4): 1,
        (3 + 4, 0, 3 + 4): 1,
        (1 + 4, 2, 3 + 4): 1,
        (2 + 4, 1, 3 + 4): -1,
        (2 + 4, 3, 1 + 4): 1,
        (3 + 4, 2, 1 + 4): -1,
        (3 + 4, 1, 2 + 4): 1,
        (1 + 4, 3, 2 + 4): -1,
    }

    _conj_coefs = {
        (0, 0): 1,
        (1, 1): -1,
        (2, 2): -1,
        (3, 3): -1,
        (0 + 4, 0 + 4): 1,
        (1 + 4, 1 + 4): -1,
        (2 + 4, 2 + 4): -1,
        (3 + 4, 3 + 4): -1,
    }

    _conj_coefs_e = {
        (0, 0): 1,
        (1, 1): -1,
        (2, 2): -1,
        (3, 3): -1,
        (0 + 4, 0 + 4): -1,
        (1 + 4, 1 + 4): 1,
        (2 + 4, 2 + 4): 1,
        (3 + 4, 3 + 4): 1,
    }

    @property
    def conj(self):
        res = self.copy()
        for i, j in self._conj_coefs:
            res.data[..., i] = self._conj_coefs[(i, j)] * res.data[..., j]
        return res

    @property
    def conj_e(self):
        res = self.copy()
        for i, j in self._conj_coefs_e:
            res.data[..., i] = self._conj_coefs_e[(i, j)] * self.data[..., j]
        return res

    def inv(self):
        temp = self * self.conj
        res = self.conj * temp.conj_e
        res.data[...] /= self.norm4()[..., np.newaxis]
        return res

    def normalized(self):
        return self / self.norm()

    def norm(self):
        return np.sqrt(self.norm2())

    def norm2(self):
        return np.sqrt(self.norm4())

    def norm4(self):
        res = self * self.conj
        res = res * res.conj_e
        return res.data[..., 0]

    @classmethod
    def translate(cls, xs: np.ndarray):
        shape = np.array(xs.data.shape)
        shape[-1] = 8
        res_data = np.zeros(shape, dtype=xs.dtype)
        res_data[..., 0] = 1
        res_data[..., 5:] = xs
        return cls(res_data)

    def transform(self, xs: Union[np.ndarray, "ndarray"]):
        if isinstance(xs, np.ndarray):
            xs_dq = ndarray.translate(xs)
            return (self * xs_dq * self.conj_e).data[..., 5:]
        elif isinstance(xs, ndarray):
            return self * xs * self.conj_e
        else:
            raise NotImplementedError


def rotx(theta):
    return ndarray([np.cos(theta / 2), np.sin(theta / 2), 0, 0, 0, 0, 0, 0])


def roty(theta):
    return ndarray([np.cos(theta / 2), 0, np.sin(theta / 2), 0, 0, 0, 0, 0])


def rotz(theta):
    return ndarray([np.cos(theta / 2), 0, 0, np.sin(theta / 2), 0, 0, 0, 0])


def transx(d):
    return ndarray([1, 0, 0, 0, 0, d / 2, 0, 0])


def transy(d):
    return ndarray([1, 0, 0, 0, 0, 0, d / 2, 0])


def transz(d):
    return ndarray([1, 0, 0, 0, 0, 0, 0, d / 2])


identity: Final[ndarray] = constant(ndarray([1, 0, 0, 0, 0, 0, 0, 0]))
i: Final[ndarray] = constant(ndarray([0, 1, 0, 0, 0, 0, 0, 0]))
j: Final[ndarray] = constant(ndarray([0, 0, 1, 0, 0, 0, 0, 0]))
k: Final[ndarray] = constant(ndarray([0, 0, 0, 1, 0, 0, 0, 0]))
e: Final[ndarray] = constant(ndarray([0, 0, 0, 0, 1, 0, 0, 0]))
