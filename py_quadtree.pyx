from libc.stdint cimport *
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
import numpy as np
cimport numpy as np

cdef extern from "quadtree.c":

    ctypedef uint64_t qt_Zpoint

    ctypedef struct qt_Tree:
        size_t length
        size_t allocated_space
        qt_Zpoint *buffer

    int qt_init(qt_Tree *tree)
    void qt_free(qt_Tree tree)

    qt_Zpoint qt_zpoint(uint32_t x, uint32_t y)
    void qt_zpoint_decode(qt_Zpoint z, uint32_t *x, uint32_t *y)

    ssize_t qt_lookup(qt_Tree tree, uint32_t x, uint32_t y)

    int qt_zinsert_multi(qt_Tree *tree, size_t inp_length, qt_Zpoint *inp)

    int qt_point_radius(qt_Tree tree,
            uint32_t center_x, uint32_t center_y, double radius,
            uint32_t **res_xs, uint32_t **res_ys, size_t *res_length);

    void bucket_sort(size_t buffer_size, uint64_t *buffer, uint64_t *scratch_buffer)

def radix_sort(np.ndarray[np.uint64_t, ndim=1, mode='c'] inp):
    cdef length = inp.shape[0]
    cdef uint64_t *scratch = <uint64_t *> malloc(sizeof(uint64_t) * length)
    if scratch == NULL:
        raise MemoryError
    bucket_sort(length, &inp[0], scratch)
    free(scratch)

def pack_zpoints(
        np.ndarray[np.uint32_t, ndim=1, mode='c'] xs,
        np.ndarray[np.uint32_t, ndim=1, mode='c'] ys
        ):

    assert xs.shape[0] == ys.shape[0]
    cdef size_t size = xs.shape[0]
    cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] res = np.empty(
            (xs.shape[0],), dtype=np.uint64)

    cdef size_t i
    for i in range(size):
        res[i] = qt_zpoint(xs[i], ys[i])

    return res

def unpack_zpoints(np.ndarray[np.uint64_t, ndim=1, mode='c'] points):
    cdef size_t length = points.shape[0]
    cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] xs = np.zeros((length,), dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] ys = np.zeros((length,), dtype=np.uint32)

    cdef uint32_t x
    cdef uint32_t y

    cdef size_t i
    for i in range(length):
        qt_zpoint_decode(points[i], &x, &y)
        xs[i] = x
        ys[i] = y

    return (xs, ys)



cdef class Quadtree:

    cdef qt_Tree tree

    def __cinit__(self):
        if qt_init(&self.tree) < 0:
            raise MemoryError

    def __dealloc__(self):
        qt_free(self.tree)

    def index(self, uint32_t x, uint32_t y):
        cdef ssize_t res = qt_lookup(self.tree, x, y)
        if res < 0:
            raise IndexError
        else:
            return res

    def indexes(self,
            np.ndarray[np.uint32_t, ndim=1, mode='c'] xs,
            np.ndarray[np.uint32_t, ndim=1, mode='c'] ys):

        assert xs.shape[0] == ys.shape[0]

        cdef np.ndarray[np.int64_t, ndim=1, mode='c'] idxes = np.zeros(
                (xs.shape[0],), dtype=np.int64)

        cdef uint32_t x
        cdef uint32_t y
        cdef ssize_t i = xs.shape[0] - 1
        while i >= 0:
            x = xs[i]
            y = ys[i]
            idxes[i] = <int64_t> qt_lookup(self.tree, x, y)
            i -= 1

        return idxes


    def insert_multi(self,
        np.ndarray[np.uint32_t, ndim=1, mode='c'] xs,
        np.ndarray[np.uint32_t, ndim=1, mode='c'] ys):

        assert xs.shape[0] == ys.shape[0]
        cdef size_t length = xs.shape[0]

        cdef qt_Zpoint *points = <qt_Zpoint *> malloc(sizeof(qt_Zpoint) * length)
        if points == NULL:
            raise MemoryError

        cdef size_t i
        for i in range(length):
            points[i] = qt_zpoint(xs[i], ys[i])

        cdef int res = qt_zinsert_multi(&self.tree, length, points)

        free(points)

        if res < 0:
            raise MemoryError

    def __len__(self):
        return self.tree.length

    def get_buffer(self):
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] res = np.empty(
                (self.tree.length,), dtype=np.uint64)
        memcpy(&res[0], self.tree.buffer, sizeof(uint64_t) * self.tree.length)
        return res

    def point_radius(self, x, y, radius):
        cdef uint32_t *res_xs
        cdef uint32_t *res_ys
        cdef uint32_t center_x = x
        cdef uint32_t center_y = y
        cdef double r = radius
        cdef size_t res_length
        cdef int retval = qt_point_radius(self.tree,
                center_x, center_y, r,
                &res_xs, &res_ys, &res_length)
        if retval == -1:
            raise MemoryError
        elif retval < 0:
            raise Exception

        if res_length < 1:
            return (np.array([]), np.array([]))

        cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] arr_xs = np.empty(
                (res_length,), dtype=np.uint32)
        cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] arr_ys = np.empty(
                (res_length,), dtype=np.uint32)

        memcpy(&arr_xs[0], res_xs, sizeof(uint32_t) * res_length)
        memcpy(&arr_ys[0], res_ys, sizeof(uint32_t) * res_length)

        free(res_xs)
        free(res_ys)

        return (arr_xs, arr_ys)
