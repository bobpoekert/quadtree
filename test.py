import unittest
import numpy as np
import py_quadtree as qt

class TestZcurve(unittest.TestCase):


    def test_roundtrip(self):
        xs = np.random.randint(2**32, size=(10**4,), dtype=np.uint32)
        ys = np.random.randint(2**32, size=(10**4,), dtype=np.uint32)

        points = qt.pack_zpoints(xs, ys)

        res_xs, res_ys = qt.unpack_zpoints(points)

        print(res_xs[:100] - xs[:100])

        self.assertEqual(np.count_nonzero(res_xs - xs), 0)
        self.assertEqual(np.count_nonzero(res_ys - ys), 0)

if __name__ == '__main__':
    unittest.main()
