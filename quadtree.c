#pragma once

#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "quadtree.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

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
inline qt_Zpoint qt_zpoint(uint32_t x, uint32_t y) {
    uint64_t x64 = (uint64_t) x;
    uint64_t y64 = (uint64_t) y;
    x64 = (x64 | (x64 << 16)) & 0x0000FFFF0000FFFF;
    x64 = (x64 | (x64 << 8)) & 0x00FF00FF00FF00FF;
    x64 = (x64 | (x64 << 4)) & 0x0F0F0F0F0F0F0F0F;
    x64 = (x64 | (x64 << 2)) & 0x3333333333333333;
    x64 = (x64 | (x64 << 1)) & 0x5555555555555555;

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

inline void qt_zpoint_decode(qt_Zpoint z, uint32_t *x, uint32_t *y) {
    *x = morton_1(z);
    *y = morton_1(z >> 1);
}

#endif

int qt_init(qt_Tree *tree) {
    tree->length = 0;
    tree->allocated_length = 0;
    tree->buffer = NULL;
    return 0;
}

void qt_free(qt_Tree tree) {
    free(tree.buffer);
}

size_t qt_zlookup(qt_Tree tree, qt_Zpoint target) {
    /* binary search */
    size_t left = 0;
    size_t right = tree.length - 1;
    size_t cursor;

    while(left <= right) {
        cursor = (left + right) / 2;
        qt_Zpoint pivot = tree.buffer[cursor];
        if (pivot < target) {
            left = cursor + 1;
        } else if (pivot > target) {
            right = cursor - 1;
        } else {
            return cursor;
        }

    }

    return cursor;
}

ssize_t qt_lookup(qt_Tree tree, uint32_t x, uint32_t y) {
    if (tree.length < 1) return -1;
    qt_Zpoint target = qt_zpoint(x, y);
    size_t res = qt_zlookup(tree, target);
    if (tree.buffer[res] == target) {
        return (ssize_t) res; 
    } else {
        return -1;
    }
}

int qt_extend(qt_Tree *tree) {
    tree->allocated_length *= 2;
    tree->buffer = realloc(tree->buffer, tree->allocated_length);
    return tree->buffer == 0 ? -1 : 0;
}

ssize_t qt_zinsert(qt_Tree *tree, qt_Zpoint target) {
    if (tree->allocated_length < 1) {
        tree->buffer = malloc(sizeof(qt_Zpoint) * QT_INIT_BUFFER_SIZE);
        if (tree->buffer == NULL) return -1;
        tree->allocated_length = QT_INIT_BUFFER_SIZE;
    }

    size_t pivot = qt_zlookup(*tree, target);
    if (tree->buffer[pivot] == target) return pivot;

    if (tree->allocated_length <= tree->length+1) {
        if (qt_extend(tree) < 0) return -1;
    }

    memmove(&tree->buffer[pivot + 1], &tree->buffer[pivot], tree->length - pivot);

    tree->length++;
    tree->buffer[pivot] = target;
    return 0;
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
            res[cursor_res] = v_b;
            cursor_res++;
            cursor_b++;
        } else if (v_a < v_b) {
            res[cursor_res] = v_a;
            cursor_res++;
            cursor_a++;
        } else  {
            res[cursor_res] = v_a;
            cursor_res++;
            cursor_a++;
            cursor_b++;
        }
        cursor_inp++;
    }

    return cursor_res;

}

int qt_zinsert_multi(qt_Tree *tree, size_t inp_length, qt_Zpoint *inp) {
    // bucket sort inp, then riffle merge into tree->buffer
    if (inp_length < 1) return 0;
    uint64_t *scratch_buffer = malloc(sizeof(uint64_t) * inp_length);
    if (!scratch_buffer) return -1;


    bucket_sort(inp_length, inp, scratch_buffer);

    if(tree->length > 0) {
        size_t res_length = inp_length + tree->length;
        uint64_t *res = malloc(sizeof(uint64_t) * res_length);
        if (!res) {
            free(scratch_buffer);
            return -1;
        }
        res_length = riffle_merge(tree->length, tree->buffer, inp_length, inp, res);
        res = realloc(res, res_length * sizeof(uint64_t));
        tree->length = res_length;
        tree->allocated_length = res_length;
        uint64_t *old_buffer = tree->buffer;
        tree->buffer = res;
        free(old_buffer);
        return 0;
    } else {
        if (tree->allocated_length > 0) free(tree->buffer);
        tree->buffer = malloc(sizeof(qt_Zpoint) * inp_length);
        memcpy(tree->buffer, inp, sizeof(qt_Zpoint) * inp_length);
        tree->length = inp_length;
        tree->allocated_length = inp_length;
        return 0;
    }
}


void qt_points(qt_Tree tree, uint32_t **res_xs, uint32_t **res_ys) {

    *res_xs = malloc(sizeof(uint32_t) * tree.length);
    *res_ys = malloc(sizeof(uint32_t) * tree.length);

    if (!(*res_xs) || !(*res_ys)) return;

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

inline size_t qt_longest_common_prefix(qt_Zpoint a, qt_Zpoint b, qt_Zpoint *res_mask) {
    uint64_t mask = -1;
    size_t length = 0;
    while (length < 64 && (a & mask) != (b & mask)) {
        length++;
        mask <<= 1;
    }
    *res_mask = mask;
    return length;
}

int qt_point_radius(qt_Tree tree,
    uint32_t center_x, uint32_t center_y, uint32_t radius,
    uint32_t **res_xs, uint32_t **res_ys, size_t *res_length) {

    // outer bbox is the smallest square our circle fits inside
    // inner bbox is the largest square that fits inside our circle
    // we only have to actually calculate distance for points that lie within outer minus inner

    uint32_t outer_bbox_size = radius * 2;
    uint32_t inner_bbox_size = (uint32_t) (sqrt(2 * radius) / 2);

    qt_Zpoint outer_point_a = qt_zpoint(center_x - outer_bbox_size, center_y - outer_bbox_size);
    qt_Zpoint outer_point_b = qt_zpoint(center_x + outer_bbox_size, center_y - outer_bbox_size);
    qt_Zpoint outer_point_c = qt_zpoint(center_x + outer_bbox_size, center_y + outer_bbox_size);
    qt_Zpoint outer_point_d = qt_zpoint(center_x - outer_bbox_size, center_y + outer_bbox_size);
    
    size_t outer_inter_prefix_length = 0;
    uint64_t outer_inter_prefix_mask = -1;
    uint64_t local_prefix_mask;
    size_t local_prefix_length;

    // longest common prefix of morton codes is lowest common ancestor in a quadtree
    local_prefix_length = qt_longest_common_prefix(outer_point_a, outer_point_b, &local_prefix_mask);
    outer_inter_prefix_length = MIN(outer_inter_prefix_length, local_prefix_length);
    outer_inter_prefix_mask &= local_prefix_mask;
    local_prefix_length = qt_longest_common_prefix(outer_point_a, outer_point_c, &local_prefix_mask);
    outer_inter_prefix_length = MIN(outer_inter_prefix_length, local_prefix_length);
    outer_inter_prefix_mask &= local_prefix_mask;
    local_prefix_length = qt_longest_common_prefix(outer_point_a, outer_point_d, &local_prefix_mask);
    outer_inter_prefix_length = MIN(outer_inter_prefix_length, local_prefix_length);
    outer_inter_prefix_mask &= local_prefix_mask;
    local_prefix_length = qt_longest_common_prefix(outer_point_b, outer_point_c, &local_prefix_mask);
    outer_inter_prefix_length = MIN(outer_inter_prefix_length, local_prefix_length);
    outer_inter_prefix_mask &= local_prefix_mask;
    local_prefix_length = qt_longest_common_prefix(outer_point_b, outer_point_d, &local_prefix_mask);
    outer_inter_prefix_length = MIN(outer_inter_prefix_length, local_prefix_length);
    outer_inter_prefix_mask &= local_prefix_mask;
    local_prefix_length = qt_longest_common_prefix(outer_point_c, outer_point_d, &local_prefix_mask);
    outer_inter_prefix_length = MIN(outer_inter_prefix_length, local_prefix_length);
    outer_inter_prefix_mask &= local_prefix_mask;

    qt_Zpoint outer_point_min = outer_point_a & outer_inter_prefix_mask;
    qt_Zpoint outer_point_max = outer_point_a | ~outer_inter_prefix_mask;

    // all of our result points lie within this range
    size_t outer_min_idx = qt_zlookup(tree, outer_point_min);
    size_t outer_max_idx = qt_zlookup(tree, outer_point_max);

    uint32_t *res_xs_buf = malloc(sizeof(uint32_t) * (outer_max_idx - outer_min_idx));
    if (!res_xs_buf) return -1;
    uint32_t *res_ys_buf = malloc(sizeof(uint32_t) * (outer_max_idx - outer_min_idx));
    if (!res_ys_buf) {
        free(res_xs_buf);
        return -1;
    }

    uint32_t inner_max_x = center_x + inner_bbox_size;
    uint32_t inner_min_x = center_x - inner_bbox_size;
    uint32_t inner_max_y = center_y + inner_bbox_size;
    uint32_t inner_min_y = center_y - inner_bbox_size;

    size_t _res_length = 0;
    size_t buf_cursor = outer_min_idx;
    for (; buf_cursor < outer_max_idx; buf_cursor++) {
        uint32_t x, y;
        int64_t dx, dy;
        qt_Zpoint inp = tree.buffer[buf_cursor];
        qt_zpoint_decode(inp, &x, &y);
        dx = x - center_x;
        dy = y - center_y;

        if ((x >= inner_min_x && x <= inner_max_x && y >= inner_min_y && y <= inner_max_y) ||
            (sqrt(dx*dx + dy*dy) <= radius)) {

            res_xs_buf[_res_length] = x;
            res_ys_buf[_res_length] = y;
            _res_length++;

        }

    }

    res_xs_buf = realloc(res_xs_buf, sizeof(uint32_t) * _res_length);
    res_ys_buf = realloc(res_ys_buf, sizeof(uint32_t) * _res_length);

    *res_xs = res_xs_buf;
    *res_ys = res_ys_buf;
    *res_length = _res_length;
    
    return 0;

}
