import numpy as np
import py_quadtree as qt
import time

size = 10**8

test_data = np.random.randint(2**64, size=(size,), dtype=np.uint64)

npy_start = time.time()
np.sort(test_data)
npy_end = time.time()
tn = npy_end - npy_start
print('numpy time: %f' % (npy_end - npy_start))

start = time.time()
qt.radix_sort(test_data)
end = time.time()
tq = end - start
print('qt time: %f' % (end - start))

print('%f%%' % ((tn - tq) / tn))
