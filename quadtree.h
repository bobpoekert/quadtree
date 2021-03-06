
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

#define QT_INIT_BUFFER_SIZE 4096

typedef uint64_t qt_Zpoint;

typedef struct qt_Tree {
    size_t length;
    size_t allocated_length;
    qt_Zpoint *buffer;
} qt_Tree;

/* zero indicates success, -1 indicates out of memory */
int qt_init(qt_Tree *tree);

void qt_free(qt_Tree tree);

/* compute Z-order curve value for coords */
qt_Zpoint qt_zpoint(uint32_t x, uint32_t y);
void qt_zpoint_decode(qt_Zpoint z, uint32_t *x, uint32_t *y);

/* lookup point. returns index into tree buffer */
size_t qt_zlookup(qt_Tree tree, qt_Zpoint targat);
/* lookup point. returns index into tree buffer, or -1 if not found */
ssize_t qt_lookup(qt_Tree tree, uint32_t x, uint32_t y);

/* insert an array of points. significantly faster than inserting points one at a time
 (inserting points one at a time is literally insertion sort here)

 return value: 0 for success, -1 for memory error
 */
int qt_zinsert_multi(qt_Tree *tree, size_t inp_length, qt_Zpoint *inp);
int qt_insert_multi(qt_Tree *tree, size_t inp_length, uint32_t *xs, uint32_t *ys);

/* writes the x and y coordinates of points in tree to malloc'd buffers assigned to xs and ys */
void qt_points(qt_Tree tree, uint32_t **res_xs, uint32_t **res_ys);

/* finds all the points within radius of the given center point 
 returns zero on success and a negative value on error.
 -1 indicates out of memory
 */
int qt_point_radius(qt_Tree tree,
    uint32_t center_x, uint32_t center_y, double radius,
    uint32_t **res_xs, uint32_t **res_ys, size_t *res_length);