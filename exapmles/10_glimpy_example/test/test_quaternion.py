from modules.quaternion import ndarray as q
from modules.quaternion import i, j, k, identity

import numpy as np


def test_quaternion():
    # shape
    a = q.zeros((1, 2, 3))
    assert a.shape == (1, 2, 3)
    # basis
    assert i * j == k
    # conjugate
    assert i.conj == -i
    # add
    assert i + j == q([0, 1, 1, 0])
    # multiply
    a = i
    a *= i
    assert a == -1
    assert i * 2 == q([0, 2, 0, 0])
    assert 2 * i == q([0, 2, 0, 0])
    a = q([[1, 0, 0, 0], [1, 0, 0, 0]])
    assert np.all(a * a == a)
    a = q.zeros((1, 2, 3))
    assert (a * a).shape == (1, 2, 3)
    # inv
    assert 1 / i == -i
    assert i / i == 1
    assert identity / identity == 1
    a = q([[1, 0, 0, 0], [1, 0, 0, 0]])
    assert np.all(a / a == a)

    # add scalar
    assert i + 1 == q([1, 1, 0, 0])
    assert 1 + i == q([1, 1, 0, 0])
    # subtract scalar
    assert i - 1 == q([-1, 1, 0, 0])
    assert 1 - i == q([1, -1, 0, 0])
    # norm
    a = q([1, 0, 0, 0])
    assert a.norm2() == 1
    a = q([[1, 0, 0, 0], [1, 0, 0, 0]])
    assert np.all(a.norm2() == np.array([1, 1]))
    assert np.all(a / a.norm2() == a)
    a[0] = 2
    assert np.all(a == q([[2, 0, 0, 0], [1, 0, 0, 0]]))

    # constant
    a = i
    a = a.astype(np.float32)
    a += 2
    assert i == q([0, 1, 0, 0])
    a = i
    a = 2
    assert i == q([0, 1, 0, 0])

    # inplace
    a = i
    a += j
    assert i == q([0, 1, 0, 0])
    assert a == q([0, 1, 1, 0])
    a = i
    a *= j
    assert a == q([0, 0, 0, 1])
    a = i
    a /= j
    assert a == q([0, 0, 0, -1])

    # set get
    a = q.zeros([2, 2])
    a[0, 1] = i
    b = q([[[0, 0, 0, 0], [0, 1, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0]]])
    assert np.all(a == b)

    # list
    a = q.array([0, i])
    assert np.all(a == q([[0, 0, 0, 0], [0, 1, 0, 0]]))
