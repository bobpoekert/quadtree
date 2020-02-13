import unittest
import numpy as np
import py_quadtree as qt
from random import randint

def naive_interleave(xs, ys):
    assert xs.shape == ys.shape
    assert xs.dtype == np.uint32
    assert ys.dtype == np.uint32
    res = np.zeros(xs.shape, dtype=np.uint64)

    xs = xs.astype(np.uint64)
    ys = ys.astype(np.uint64)

    for i in range(32):
        bit_mask = 1 << i
        x_i = (xs & bit_mask) >> i
        y_i = (ys & bit_mask) >> i
        res = res | (x_i << (2 * i))
        res = res | (y_i << (2 * i + 1))

    return res

def naive_deinterleave(vs):
    xs = np.zeros(vs.shape, dtype=np.uint32)
    ys = np.zeros(vs.shape, dtype=np.uint32)

    for i in range(32):
        xs = xs | (((vs & (1 << (2 * i))) >> (2 * i)) << i)
        ys = ys | (((vs & (1 << (2 * i + 1))) >> (2 * i + 1)) << i)

    return xs, ys

test_data = (
            np.random.randint(2**32, size=(10**5,), dtype=np.uint32),
            np.random.randint(2**32, size=(10**5,), dtype=np.uint32)
            )

test_data = {}

def generate_points(_min=0, _max=2**32):
    if (_min, _max) not in test_data:
        test_data[(_min, _max)] = (
            np.random.randint(_min, _max, size=(10**5,), dtype=np.uint32),
            np.random.randint(_min, _max, size=(10**5,), dtype=np.uint32)
            )
    return test_data[(_min, _max)]

class TestCase(unittest.TestCase):

    def assertArrayZero(self, arr):
        self.assertEqual(np.count_nonzero(arr), 0)

class TestZcurve(TestCase):

    def test_naive_roundtrip(self):
        xs, ys = generate_points()

        packed = naive_interleave(xs, ys)
        xs_p, ys_p = naive_deinterleave(packed)

        self.assertArrayZero(xs_p - xs)
        self.assertArrayZero(ys_p - ys)

    def test_naive(self):
        xs, ys = generate_points()

        n_res = naive_interleave(xs, ys)
        res = qt.pack_zpoints(xs, ys)

        self.assertEqual(n_res.dtype, res.dtype)
        self.assertEqual(n_res.shape, res.shape)

        self.assertArrayZero(n_res - res)

    def test_naive_unpack(self):
        xs, ys = generate_points()

        packed = qt.pack_zpoints(xs, ys)
        r_xs, r_ys = naive_deinterleave(packed)

        self.assertArrayZero(r_xs - xs)
        self.assertArrayZero(r_ys - ys)

    def test_naive_pack(self):
        xs, ys = generate_points()

        packed = naive_interleave(xs, ys)
        r_xs, r_ys = qt.unpack_zpoints(packed)

        self.assertArrayZero(r_xs - xs)
        self.assertArrayZero(r_ys - ys)

    def test_roundtrip(self):
        xs, ys = generate_points()

        points = qt.pack_zpoints(xs, ys)

        res_xs, res_ys = qt.unpack_zpoints(points)

        self.assertEqual(xs.dtype, res_xs.dtype)
        self.assertEqual(xs.shape, res_xs.shape)

        self.assertArrayZero(res_xs - xs)
        self.assertArrayZero(res_ys - ys)

class TestSort(TestCase):

    def test_init(self):
        xs, ys = generate_points()

        zs = qt.pack_zpoints(xs, ys)

        tree = qt.Quadtree()
        tree.insert_multi(xs, ys)

        buf = tree.get_buffer()

        self.assertEqual(buf.shape, zs.shape)
        self.assertArrayZero(buf[:-1] > buf[1:])
        self.assertArrayZero(np.sort(zs) - buf)

    def test_lookup(self):
        xs, ys = generate_points()
        xs = xs - 1
        ys = ys - 1

        zs = qt.pack_zpoints(xs, ys)

        tree = qt.Quadtree()
        tree.insert_multi(xs, ys)

        buf = tree.get_buffer()

        res_idxes = tree.indexes(xs, ys)

        self.assertArrayZero(buf[res_idxes] - zs)
        self.assertArrayZero(np.sort(zs)[res_idxes] - zs)

        with self.assertRaises(IndexError):
            bad_idx = tree.index(2**32-1, 2**32-1)


class TestPointRadius(TestCase):

    def test_point_radius(self):

        xs, ys = generate_points(0, 2*17)

        tree = qt.Quadtree()
        tree.insert_multi(xs, ys)

        while 1:
            center_x = randint(np.min(xs), np.max(xs))
            center_y = randint(np.min(ys), np.max(ys))

            radius = (np.max(xs) - np.min(xs)) / 50

            distances = np.sqrt((xs - center_x)**2 + (ys - center_y)**2)
            targets = distances <= radius
            target_xs = xs[targets]
            target_ys = ys[targets]


            res_xs, res_ys = tree.point_radius(center_x, center_y, radius)

            print(res_xs.shape[0] - target_xs.shape[0],
                    res_ys.shape[0] - target_ys.shape[0])

            self.assertArrayZero(np.sort(res_xs) - np.sort(target_xs))
            self.assertArrayZero(np.sort(res_ys) - np.sort(target_ys))

            if res_xs.shape[0] > 0:
                break

if __name__ == '__main__':
    unittest.main()
