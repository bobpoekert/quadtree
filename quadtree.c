#pragma once

#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

#include "quadtree.h"

/* based on http://graphics.stanford.edu/~seander/bithacks.html#InterleaveBMN */
static const uint64_t B[] = {
    0x5555555555555555, 0x3333333333333333, 0x0F0F0F0F0f0f0f0f, 0x00FF00FF00ff00ff,
    0x0000ffff0000ffff
    };
static const uint64_t S[] = {1, 2, 4, 8, 16};

qt_Zpoint qt_zpoint(uint32_t x, uint32_t y) {
    // Interleave lower 32 bits of x and y, so the bits of x
    // are in the even positions and bits from y in the odd;
    uint64_t x64 = x; 
    uint64_t y64 = y; 

    x64 = (x64 | (x64 << S[4])) & B[4];
    x64 = (x64 | (x64 << S[3])) & B[3];
    x64 = (x64 | (x64 << S[2])) & B[2];
    x64 = (x64 | (x64 << S[1])) & B[1];
    x64 = (x64 | (x64 << S[0])) & B[0];

    y64 = (y64 | (y64 << S[4])) & B[4];
    y64 = (y64 | (y64 << S[3])) & B[3];
    y64 = (y64 | (y64 << S[2])) & B[2];
    y64 = (y64 | (y64 << S[1])) & B[1];
    y64 = (y64 | (y64 << S[0])) & B[0];

    return x64 | (y64 << 1);
}

void qt_zpoint_decode(qt_Zpoint z, uint32_t *x, uint32_t *y) {

    uint64_t x64 = 0;
    uint64_t y64 = 0;
    
    x64 = (z | (z << S[0])) & B[0];
    x64 = (x64 | (x64 << S[1])) & B[1];
    x64 = (x64 | (x64 << S[2])) & B[2];
    x64 = (x64 | (x64 << S[3])) & B[3];
    x64 = (x64 | (x64 << S[4])) & B[4];

    y64 = ((z >> 1) | ((z >> 1) << S[0])) & B[0];
    y64 = (y64 | (y64 << S[1])) & B[1];
    y64 = (y64 | (y64 << S[2])) & B[2];
    y64 = (y64 | (y64 << S[3])) & B[3];
    y64 = (y64 | (y64 << S[4])) & B[4];

    *x = (uint32_t) x64;
    *y = (uint32_t) y64;

}

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

    while(right > left) {
        size_t cursor = right - ((right - left) / 2);
        qt_Zpoint pivot = target.buffer[cursor];
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
    return tree.buffer[res] == target ? res : -1;
}

int qt_extend(qt_Tree *tree) {
    tree->allocated_size *= 2;
    tree->buffer = realloc(tree->buffer, tree->allocated_size);
    return tree->buffer == 0 ? -1 : 0;
}

ssize_t qt_zinsert(qt_Tree *tree, qt_Zpoint target) {
    size_t pivot = qt_zlookup(tree, target);
    if (tree.buffer[pivot] == target) return pivot;

    if (tree.allocated_size <= tree.length+1) {
        if (qt_extend(qt_Tree *tree) < 0) return -1;
    }

    memmove(&tree->buffer[pivot + 1], &tree->buffer[pivot], tree->length - pivot);

    tree->length++;
    tree->buffer[pivot] = target;
}

ssize_t qt_insert(qt_Tree *tree, uint32_t x, uint32_t y) {
    return qt_zinsert(tree, zt_zpoint(x, y));
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
        bucket_offsets = {0};
        bucket_lengths = {0};
        bucket_counts = {0};
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
    size_t cursr_inp = 0;

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

int qt_zinsert_multi(qt_Tree *tree, size_t inp_length, zt_Zpoint *inp) {
    // bucket sort inp, then riffle merge into tree->buffer
    uint64_t *scratch_buffer = malloc(sizeof(uint64_t) * inp_length);
    if (!scratch_buffer) return -1;

    bucket_sort(inp_length, inp, scratch_buffer);

    size_t res_length = inp_length + tree->length;
    uint64_t *res = malloc(sizeof(uint64_t) * res_length);
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