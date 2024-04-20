from typing import Final

import numpy as np
from modules.ndarray import constant
from modules.ndarray import ndarray as _ndarray


class ndarray(_ndarray):
    dim = 4
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
    }
    _conj_coefs = {
        (0, 0): 1,
        (1, 1): -1,
        (2, 2): -1,
        (3, 3): -1,
    }

    @property
    def conj(self):
        conj = self.copy()
        for i, j in self._conj_coefs:
            conj.data[..., i] = self._conj_coefs[(i, j)] * conj.data[..., j]
        return conj

    def inv(self):
        res = self.conj
        res.data[...] = res.data[...] / self.norm2()[..., np.newaxis]
        return res

    def normalized(self):
        return self / self.norm()

    def norm(self):
        return np.sqrt(self.norm2())

    def norm2(self):
        res = self * self.conj
        return res.data[..., 0]


identity: Final[ndarray] = constant(ndarray([1, 0, 0, 0]))
i: Final[ndarray] = constant(ndarray([0, 1, 0, 0]))
j: Final[ndarray] = constant(ndarray([0, 0, 1, 0]))
k: Final[ndarray] = constant(ndarray([0, 0, 0, 1]))
