#ifndef ARROW_BRIDGE_H
#define ARROW_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int dim;           /* expected 4 */
    int vertex_count;  /* number of vertices */
    long *coords;      /* flat array: vertex_count * dim */
} KSRow;

typedef int (*ks_row_callback)(const KSRow *row, void *user_data);

/* Scan all .arrow IPC files in a directory and invoke callback per row.
   Returns 0 on success, nonzero on error. */
int ks_scan_arrow_dir(const char *dataset_dir,
                      ks_row_callback cb,
                      void *user_data);

#ifdef __cplusplus
}
#endif

#endif