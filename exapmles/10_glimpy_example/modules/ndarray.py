from typing import Any, Dict, List, Tuple, Union
import numpy as np


class ndarray:
    dim: int
    _mul_coefs: Dict[Tuple[int, int, int], float]

    def __init__(self, data: Union[np.ndarray, List]):
        self.is_constant = False
        if isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, list):
            self.data = np.array(data)
        else:
            print(f"{type(data) = }")
            raise NotImplementedError
        assert (
            self.data.shape[-1] == self.__class__.dim
        ), f"{self.data.shape = }"

    def tolist(self) -> tuple:
        return self.data.tolist()

    @property
    def shape(self) -> tuple:
        return self.data.shape[:-1]

    def astype(self, dtype):
        data = self.data.astype(dtype=dtype)
        return self.__class__(data)

    def copy(self):
        return self.__class__(self.data.copy())

    def __neg__(self):
        return self.__class__(-self.data)

    def __mul__(self, other: Any):
        other = self.__class__.array(other)
        shape = (
            self.shape if len(self.shape) > len(other.shape) else other.shape
        )
        res = self.__class__.zeros(shape)
        for i, j, k in self._mul_coefs:
            res.data[..., k] += (
                self._mul_coefs[(i, j, k)]
                * self.data[..., i]
                * other.data[..., j]
            )
        return res

    def __rmul__(self, other: Any):
        return self * self.__class__.array(other)

    def inv(self):
        raise NotImplementedError

    def __truediv__(self, _other: Any):
        other = self.__class__.array(_other)
        res = self.copy()
        res *= other.inv()
        return res

    def __rtruediv__(self, other: Any):
        return self.__class__.array(other) / self

    def __add__(self, other):
        data = self.data + self.__class__.array(other).data
        return self.__class__(data)

    def __iadd__(self, _other: Any) -> "ndarray":
        other = self.__class__.array(_other)
        if self.is_constant:
            return self + self.__class__.array(other)
        else:
            self.data += self.__class__.array(other).data
            return self

    def __radd__(self, other: Any):
        return self + self.__class__.array(other)

    def __sub__(self, other: Any):
        return self.__class__(self.data - self.array(other).data)

    def __isub__(self, other):
        if self.is_constant:
            self.data = (self - other).data
            self.is_constant = False
        else:
            self.data -= self.__class__.array(other).data
        return self

    def __rsub__(self, other: Any):
        return -self + other

    def __eq__(self, other: Any) -> bool:
        return np.all(self.data == self.__class__.array(other).data, axis=-1)

    def __str__(self) -> str:
        return self.data.__str__()

    def __setitem__(self, slice, other: Any):
        self.data[slice] = self.__class__.array(other).data
        return self

    def __getitem__(self, slice):
        return self.__class__(self.data[slice])

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @classmethod
    def array(cls, obj: Any):
        if isinstance(obj, ndarray):
            return obj
        elif isinstance(obj, np.ndarray):
            data = np.zeros(obj.shape + (cls.dim,))
            data[..., 0] = obj
        elif isinstance(obj, list):
            new_obj: list[ndarray] = []
            for _obj in obj:
                _obj = cls.array(_obj)
                new_obj.append(_obj)
            data = cls.stack(new_obj).data
        elif isinstance(obj, (int, float)):
            data = np.zeros(cls.dim)
            data[0] = obj
        else:
            raise NotImplementedError

        assert data.shape[-1] == cls.dim, f"{data.shape = }"
        return cls(data)

    @classmethod
    def zeros(cls, shape: tuple):
        data = np.zeros(shape)
        return cls.array(data)

    @classmethod
    def stack(cls, objs: List["ndarray"]):
        res = np.stack([x.data for x in objs], axis=0)
        return cls(res)


def constant(obj: Any):
    obj.is_constant = True
    return obj
