import unittest
import numpy as np
import py_quadtree as qt

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

def generate_points():
    return (
            np.random.randint(2**32, size=(10**4,), dtype=np.uint32),
            np.random.randint(2**32, size=(10**4,), dtype=np.uint32)
            )

class TestZcurve(unittest.TestCase):

    def test_naive_roundtrip(self):
        xs, ys = generate_points()

        packed = naive_interleave(xs, ys)
        xs_p, ys_p = naive_deinterleave(packed)

        self.assertEqual(np.count_nonzero(xs_p - xs), 0)
        self.assertEqual(np.count_nonzero(ys_p - ys), 0)

    def test_naive(self):
        xs, ys = generate_points()

        n_res = naive_interleave(xs, ys)
        res = qt.pack_zpoints(xs, ys)

        self.assertEqual(n_res.dtype, res.dtype)
        self.assertEqual(n_res.shape, res.shape)

        self.assertEqual(np.count_nonzero(n_res - res), 0)

    def test_naive_unpack(self):
        xs, ys = generate_points()

        packed = qt.pack_zpoints(xs, ys)
        r_xs, r_ys = naive_deinterleave(packed)

        self.assertEqual(np.count_nonzero(r_xs - xs), 0)
        self.assertEqual(np.count_nonzero(r_ys - ys), 0)

    def test_naive_pack(self):
        xs, ys = generate_points()

        packed = naive_interleave(xs, ys)
        r_xs, r_ys = qt.unpack_zpoints(packed)

        self.assertEqual(np.count_nonzero(r_xs - xs), 0)
        self.assertEqual(np.count_nonzero(r_ys - ys), 0)

    def test_roundtrip(self):
        xs, ys = generate_points()

        points = qt.pack_zpoints(xs, ys)

        res_xs, res_ys = qt.unpack_zpoints(points)

        self.assertEqual(xs.dtype, res_xs.dtype)
        self.assertEqual(xs.shape, res_xs.shape)

        self.assertEqual(np.count_nonzero(res_xs - xs), 0)
        self.assertEqual(np.count_nonzero(res_ys - ys), 0)

if __name__ == '__main__':
    unittest.main()
