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

test_data = {}

def generate_points(_min=0, _max=2**32, cache=True):
    if cache is False:
        return (
            np.random.randint(_min, _max, size=(10**5,), dtype=np.uint32),
            np.random.randint(_min, _max, size=(10**5,), dtype=np.uint32)
            )


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

        self.assertTrue(zs.shape[0] > 0)

        tree = qt.Quadtree()
        tree.insert_multi(xs, ys)

        buf = tree.get_buffer()

        self.assertEqual(buf.shape, zs.shape)
        self.assertArrayZero(buf[:-1] > buf[1:]) # sorted
        self.assertArrayZero(buf[:-1] == buf[1:]) # no duplicates
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

        for i in range(100):
            xs, ys = generate_points(0, 2*17, cache=False)

            tree = qt.Quadtree()
            tree.insert_multi(xs, ys)

            for radius in range(15):

                radius = 0.5 + (0.1 * radius)

                while 1:
                    center_x = randint(np.min(xs), np.max(xs))
                    center_y = randint(np.min(ys)+1, np.max(ys))

                    distances = np.sqrt((xs - center_x)**2 + (ys - center_y)**2)
                    targets = distances <= radius
                    target_xs = xs[targets]
                    target_ys = ys[targets]

                    res_xs, res_ys = tree.point_radius(center_x, center_y, radius)

                    if res_xs.shape[0] > 0 and res_ys.shape[0] > 0:
                        res = np.transpose([res_xs, res_ys])
                        target = np.transpose([target_xs, target_ys])

                        res = np.unique(res, axis=0)
                        target = np.unique(target, axis=0)

                        self.assertArrayZero(res - target)

                        break

if __name__ == '__main__':
    unittest.main()
