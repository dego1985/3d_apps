from modules.dual_quaternion import ndarray as dq
from modules.dual_quaternion import identity, i, j, k, e

import numpy as np


def test_dual_quaternion():
    test_shape()
    test_add()
    test_sub()
    test_multiply()
    test_inv()
    test_conj()
    test_norm()
    test_setter()
    test_contant()
    test_astype()
    test_list()
    test_inplace()
    test_translate()


def test_shape():
    a = dq.zeros((1, 2, 3))
    assert a.shape == (1, 2, 3)


def test_add():
    assert i + j == dq([0, 1, 1, 0, 0, 0, 0, 0])
    assert i + 1 == dq([1, 1, 0, 0, 0, 0, 0, 0])
    assert 1 + i == dq([1, 1, 0, 0, 0, 0, 0, 0])


def test_sub():
    assert i - 1 == dq([-1, 1, 0, 0, 0, 0, 0, 0])
    assert 1 - i == dq([1, -1, 0, 0, 0, 0, 0, 0])


def test_multiply():
    # basis
    assert i * j == k
    assert e * e == 0

    a = i
    a *= i
    assert a == -1
    assert i * 2 == dq([0, 2, 0, 0, 0, 0, 0, 0])
    assert 2 * i == dq([0, 2, 0, 0, 0, 0, 0, 0])
    a = dq([[1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]])
    assert np.all(a * a == a)
    a = dq.zeros((1, 2, 3))
    assert (a * a).shape == (1, 2, 3)
    assert (1 + e) * (1 - e) == 1


def test_inv():
    assert 1 / i == -i
    assert i / i == 1
    assert identity / identity == 1
    a = dq.array([i, k])
    assert np.all(a / a == 1)
    assert (1 + e) / (1 + e) == 1
    assert (1 + e * i) / (1 + e * i) == 1
    a = 1 + e + i * e
    assert a / a == 1
    a = 1 + i + j + k + e + e * i + e * j + e * k
    assert a / a == 1


def test_conj():
    assert i.conj == -i
    assert e.conj == e
    assert e.conj_e == -e


def test_norm():
    # norm
    a = dq([1, 0, 0, 0, 0, 0, 0, 0])
    assert a.norm2() == 1
    a = dq([[1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0]])
    assert np.all(a.norm2() == np.array([1, 1]))
    assert np.all(a / a.norm2() == a)


def test_setter():
    a = dq([[1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]])
    a[0] = 2
    assert np.all(
        a == dq([[2, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]])
    )

    # setter
    a = dq.zeros([2, 2])
    a[0, 1] = i
    b = dq.array([[0, i], [0, 0]])
    assert np.all(a == b)


def test_contant():
    a = i
    a += 2
    assert i == dq([0, 1, 0, 0, 0, 0, 0, 0])


def test_astype():
    a = i
    a = a.astype(np.float32)
    assert a.dtype == np.float32


def test_inplace():
    a = i
    a += j
    assert a == i + j
    a = i
    a *= j
    assert a == k
    a = i
    a /= j
    assert a == -k


def test_list():
    a = dq.array([0, i])
    assert np.all(
        a == dq([[0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]])
    )


def test_translate():
    x = dq.array(1 + e * i)
    t = dq.array(1 + e * j / 2)
    assert t.transform(x) == dq.array(1 + e * (i + j))
    x = dq.array(1 + e * i)
    t = dq.array(1 + k)
    assert t.transform(x) == dq.array(2 + e * j * 2)
    x = np.array([0, 1, 2])
    t = dq.array(1 + e * j / 2)
    assert np.all(t.transform(x) == np.array([0, 2, 2]))
