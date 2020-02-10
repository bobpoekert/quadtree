#pragma once

#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

#include "quadtree.h"

#ifdef QT_MBMI2
#include <immintrin.h>
qt_Zpoint qt_zpoint(uint32_t x, uint32_t y) {
  return _pdep_u32(x, 0x55555555) | _pdep_u32(y,0xaaaaaaaa);
}

void qt_zpoint_decode(qt_Zpoint m, uint32_t *x, uint32_t *y) {
  *x = _pext_u64(m, 0x5555555555555555);
  *y = _pext_u64(m, 0xaaaaaaaaaaaaaaaa);
}
#else

// https://stackoverflow.com/questions/30539347/2d-morton-code-encode-decode-64bits
qt_Zpoint qt_zpoint(uint32_t x, uint32_t y) {
    uint64_t x64 = (uint64_t) x;
    uint64_t y64 = (uint64_t) y;
    x64 = (x64 | (x64 << 16)) & 0x640000FFFF0000FFFF;
    x64 = (x64 | (x64 << 8)) & 0x6400FF00FF00FF00FF;
    x64 = (x64 | (x64 << 4)) & 0x640F0F0F0F0F0F0F0F;
    x64 = (x64 | (x64 << 2)) & 0x643333333333333333;
    x64 = (x64 | (x64 << 1)) & 0x645555555555555555;

    y64 = (y64 | (y64 << 16)) & 0x0000FFFF0000FFFF;
    y64 = (y64 | (y64 << 8)) & 0x00FF00FF00FF00FF;
    y64 = (y64 | (y64 << 4)) & 0x0F0F0F0F0F0F0F0F;
    y64 = (y64 | (y64 << 2)) & 0x3333333333333333;
    y64 = (y64 | (y64 << 1)) & 0x5555555555555555;

    return x64 | (y64 << 1);
}

inline uint32_t morton_1(uint64_t x) {
    x = x & 0x5555555555555555;
    x = (x | (x >> 1))  & 0x3333333333333333;
    x = (x | (x >> 2))  & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x >> 4))  & 0x00FF00FF00FF00FF;
    x = (x | (x >> 8))  & 0x0000FFFF0000FFFF;
    x = (x | (x >> 16)) & 0x00000000FFFFFFFF;
    return (uint32_t)x;
}

void qt_zpoint_decode(qt_Zpoint z, uint32_t *x, uint32_t *y) {
    *x = morton_1(z);
    *y = morton_1(z >> 1);
}

#endif

int qt_init(qt_Tree *tree) {
    tree->length = 0;
    tree->allocated_size = QT_INIT_BUFFER_SIZE;
    tree->buffer = malloc(sizeof(qt_Zpoint) * QT_INIT_BUFFER_SIZE);
    return tree->buffer == 0 ? -1 : 0;
}

void qt_free(qt_Tree tree) {
    free(tree.buffer);
}

size_t qt_zlookup(qt_Tree tree, qt_Zpoint target) {
    /* binary search */
    size_t left = 0;
    size_t right = tree.length - 1;
    size_t cursor = 0;

    while(right > left) {
        cursor = right - ((right - left) / 2);
        qt_Zpoint pivot = tree.buffer[cursor];
        if (pivot == target) {
            return cursor;
        } else if (pivot < target) {
            left = cursor;
        } else {
            right = cursor;
        }
    }

    return cursor;
}

ssize_t qt_lookup(qt_Tree tree, uint32_t x, uint32_t y) {
    qt_Zpoint target = qt_zpoint(x, y);
    size_t res = qt_zlookup(tree, target);
    return (tree.buffer[res] == target) ? res : -1;
}

int qt_extend(qt_Tree *tree) {
    tree->allocated_size *= 2;
    tree->buffer = realloc(tree->buffer, tree->allocated_size);
    return tree->buffer == 0 ? -1 : 0;
}

ssize_t qt_zinsert(qt_Tree *tree, qt_Zpoint target) {
    size_t pivot = qt_zlookup(*tree, target);
    if (tree->buffer[pivot] == target) return pivot;

    if (tree->allocated_size <= tree->length+1) {
        if (qt_extend(tree) < 0) return -1;
    }

    memmove(&tree->buffer[pivot + 1], &tree->buffer[pivot], tree->length - pivot);

    tree->length++;
    tree->buffer[pivot] = target;
}

ssize_t qt_insert(qt_Tree *tree, uint32_t x, uint32_t y) {
    return qt_zinsert(tree, qt_zpoint(x, y));
}

void bucket_sort(size_t buffer_size, uint64_t *buffer, uint64_t *scratch_buffer) {
    size_t bucket_offsets[16];
    size_t bucket_lengths[16];
    size_t bucket_counts[16];
    uint64_t mod_mask = 15;
    size_t shift_bits = 0;

    uint64_t *front_buffer = buffer;
    uint64_t *back_buffer = scratch_buffer;

    while(shift_bits < 64) {
        memset(bucket_offsets, 0, sizeof(size_t) * 16);
        memset(bucket_lengths, 0, sizeof(size_t) * 16);
        memset(bucket_counts, 0, sizeof(size_t) * 16);
        for (size_t i=0; i < buffer_size; i++) {
            uint8_t radix = (uint8_t) ((front_buffer[i] & mod_mask) >> shift_bits);
            bucket_counts[radix]++;
        }
        for (size_t i=1; i < 16; i++) {
            bucket_offsets[i] = bucket_offsets[i - 1] + bucket_counts[i - 1];
        }
        for (size_t i=0; i < buffer_size; i++) {
            uint8_t radix = (uint8_t) ((front_buffer[i] & mod_mask) >> shift_bits);
            back_buffer[bucket_offsets[radix] + bucket_lengths[radix]] = front_buffer[i];
            bucket_lengths[radix]++;
        }
        mod_mask <<= 4;
        shift_bits += 4;
        uint64_t *s = back_buffer;
        back_buffer = front_buffer;
        front_buffer = s;
    }

    // 64 iterations is an even number so we need to swap buffers
    memcpy(back_buffer, sizeof(uint64_t) * buffer_size, front_buffer);
}

size_t riffle_merge(
    size_t source_a_size, uint64_t *source_a,
    size_t source_b_size, uint64_t *source_b,
    uint64_t *res
) {
    //NOTE: removes duplicate values

    size_t cursor_a = 0;
    size_t cursor_b = 0;
    size_t cursor_res = 0;
    size_t cursor_inp = 0;

    size_t max_res_size = source_a_size + source_b_size;

    while(cursor_inp < max_res_size) {
        uint64_t v_a = source_a[cursor_a];
        uint64_t v_b = source_b[cursor_b];
        if (v_a > v_b) {
            if (cursor_res < 1 || res[cursor_res - 1] != v_b) {
                res[cursor_res] = v_b;
                cursor_res++;
            }
            cursor_b++;
        } else if (v_a < v_b) {
            if (cursor_res < 1 || res[cursor_res - 1] != v_a) {
                res[cursor_res] = v_a;
                cursor_res++;
            }
            cursor_a++;
        } else  {
            if (cursor_res < 1 || res[cursor_res - 1] != v_a) {
                res[cursor_res] = v_a;
                cursor_res++;
            }
            cursor_a++;
            cursor_b++;
        }
        cursor_inp++;
    }

    return cursor_res;

}

int qt_zinsert_multi(qt_Tree *tree, size_t inp_length, qt_Zpoint *inp) {
    // bucket sort inp, then riffle merge into tree->buffer
    uint64_t *scratch_buffer = malloc(sizeof(uint64_t) * inp_length);
    if (!scratch_buffer) return -1;

    bucket_sort(inp_length, inp, scratch_buffer);

    size_t res_length = inp_length + tree->length;
    uint64_t *res = malloc(sizeof(uint64_t) * res_length);
    if (!res) {
        free(scratch_buffer);
        return -1;
    }
    res_length = riffle_merge(tree->length, tree->buffer, inp_length, inp, res);
    res = realloc(res, res_length * sizeof(uint64_t));
    tree->length = res_length;
    uint64_t old_buffer = tree->buffer;
    tree->buffer = res;
    free(old_buffer);
    return 0;
}

void qt_points(qt_Tree tree, uint32_t **res_xs, uint32_t **res_ys) {

    *res_xs = malloc(sizeof(uint32_t) * tree.length);
    *res_ys = malloc(sizeof(uint32_t) * tree.length);

    for (size_t i=0; i < tree.length; i++) {
        uint32_t x;
        uint32_t y;
        qt_zpoint_decode(tree.buffer[i], &x, &y);
        (*res_xs)[i] = x;
        (*res_ys)[i] = y;
    }

}

int qt_insert_multi(qt_Tree *tree, size_t input_length, uint32_t *xs, uint32_t *ys) {
    qt_Zpoint *zpoints = malloc(sizeof(qt_Zpoint) * input_length);
    if (!zpoints) return -1;
    for (size_t i=0; i < input_length; i++) {
        zpoints[i] = qt_zpoint(xs[i], ys[i]);
    }
    int res = qt_zinsert_multi(tree, input_length, zpoints);
    free(zpoints);
    return res;
}