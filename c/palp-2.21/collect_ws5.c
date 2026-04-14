#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Global.h"
#include "Subpoly.h"
#include "arrow_bridge.h"

FILE *inFILE = NULL;
FILE *outFILE = NULL;

#define TARGET_DIM      4
#define TARGET_WLEN     5
#define MAX_LOCAL_WS    256
#define MAX_SCAN_ROWS   1000000

typedef struct {
    int len;
    Long w[TARGET_WLEN];
} WS5;

typedef struct {
    long rows_seen;
    long rows_reflexive;
    long ws_written;
    long stop_after_rows;
    int  vertex_count_filter;   /* 0 = no filter */
    long vc_hist[65];
    FILE *fout;
} Stats;

/* ---------- helpers ---------- */

static Long gcd_long(Long a, Long b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b != 0) {
        Long t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static void sort_long_array(Long *a, int n) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            if (a[j] < a[i]) {
                Long t = a[i];
                a[i] = a[j];
                a[j] = t;
            }
        }
    }
}

static void normalize_ws5(WS5 *x) {
    int i;
    Long g = 0;

    for (i = 0; i < x->len; i++) {
        g = gcd_long(g, x->w[i]);
    }
    if (g <= 0) g = 1;

    for (i = 0; i < x->len; i++) x->w[i] /= g;

    sort_long_array(x->w, x->len);
}

static int ws5_equal(const WS5 *a, const WS5 *b) {
    int i;
    if (a->len != b->len) return 0;
    for (i = 0; i < a->len; i++) {
        if (a->w[i] != b->w[i]) return 0;
    }
    return 1;
}

static int vec_is_zero_long(const Long *x, int n) {
    int i;
    for (i = 0; i < n; i++) {
        if (x[i] != 0) return 0;
    }
    return 1;
}

static int vec_equal_long(const Long *a, const Long *b, int n) {
    int i;
    for (i = 0; i < n; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

static int sanitize_poly_vertices(PolyPointList *P) {
    int i, j, k;
    int out = 0;

    for (i = 0; i < P->np; i++) {
        if (vec_is_zero_long(P->x[i], P->n)) continue;

        int dup = 0;
        for (j = 0; j < out; j++) {
            if (vec_equal_long(P->x[i], P->x[j], P->n)) {
                dup = 1;
                break;
            }
        }
        if (dup) continue;

        if (out != i) {
            for (k = 0; k < P->n; k++) {
                P->x[out][k] = P->x[i][k];
            }
        }
        out++;
    }

    P->np = out;
    return (P->np >= 5 && P->np <= VERT_Nmax);
}

static int row_to_poly(const KSRow *row, PolyPointList *P, VertexNumList *V) {
    int i, j;

    if (!row || !P || !V) return 0;
    if (row->dim != TARGET_DIM) return 0;
    if (row->vertex_count <= 0 || row->vertex_count > VERT_Nmax) return 0;

    P->n = row->dim;
    P->np = row->vertex_count;

    for (i = 0; i < row->vertex_count; i++) {
        V->v[i] = i;
        for (j = 0; j < row->dim; j++) {
            P->x[i][j] = row->coords[i * row->dim + j];
        }
    }
    V->nv = row->vertex_count;
    return 1;
}

/*
  Extract 5-weight systems from the VERTICES of the dual polytope.
  This avoids the huge complete dual point set and is much stabler.
*/
static int extract_ws5_from_dual_vertices(PolyPointList *P,
                                          VertexNumList *V,
                                          EqList *E,
                                          WS5 *out,
                                          int max_out)
{
    static PolyPointList DPv;
    static Long CM[VERT_Nmax][POLY_Dmax];
    static Long Wraw[FIB_Nmax][VERT_Nmax];

    int dual_dim;
    int nw = 0;
    int p_clean = 0;
    int i, j, k;
    int count = 0;

    memset(&DPv, 0, sizeof(DPv));
    memset(CM,   0, sizeof(CM));
    memset(Wraw, 0, sizeof(Wraw));

    dual_dim = P->n;

    /*
      For reflexive P, the facet equations E correspond to the vertices of P*.
      EL_to_PPL converts the equations to the dual vertices.
    */
    if (!EL_to_PPL(E, &DPv, &dual_dim)) {
        return 0;
    }
    DPv.n = dual_dim;

    if (DPv.n != TARGET_DIM) return 0;
    if (DPv.np <= 0 || DPv.np > VERT_Nmax) return 0;

    /* remove zero vector and duplicates */
    for (i = 0; i < DPv.np; i++) {
        if (vec_is_zero_long(DPv.x[i], DPv.n)) continue;

        int dup = 0;
        for (j = 0; j < p_clean; j++) {
            if (vec_equal_long(DPv.x[i], CM[j], DPv.n)) {
                dup = 1;
                break;
            }
        }
        if (dup) continue;

        if (p_clean >= VERT_Nmax) return 0;

        for (j = 0; j < DPv.n; j++) {
            CM[p_clean][j] = DPv.x[i][j];
        }
        p_clean++;
    }

    if (p_clean < TARGET_WLEN) return 0;

    /*
      codim = 1:
      restrict to the relevant 4d simplices producing 5-weight systems
    */
    IP_Simplex_Decomp(CM, p_clean, DPv.n, &nw, Wraw, FIB_Nmax, 0);

    for (i = 0; i < nw && count < max_out; i++) {
        WS5 x;
        int nonzero = 0;

        x.len = 0;

        for (k = 0; k < VERT_Nmax; k++) {
            if (Wraw[i][k] != 0) {
                if (x.len < TARGET_WLEN) {
                    x.w[x.len++] = Wraw[i][k];
                }
                nonzero++;
            }
        }

        if (nonzero != TARGET_WLEN) continue;
        if (x.len != TARGET_WLEN) continue;

        normalize_ws5(&x);

        /* local dedup */
        {
            int seen = 0, t;
            for (t = 0; t < count; t++) {
                if (ws5_equal(&out[t], &x)) {
                    seen = 1;
                    break;
                }
            }
            if (!seen) {
                out[count++] = x;
            }
        }
    }

    return count;
}

static void write_ws5(FILE *fout, long row_id, const WS5 *x) {
    int i;
    fprintf(fout, "%ld", row_id);
    for (i = 0; i < x->len; i++) {
        fprintf(fout, " %ld", x->w[i]);
    }
    fprintf(fout, "\n");
}

/* ---------- callback ---------- */

static int process_row(const KSRow *row, void *user_data) {
    Stats *S = (Stats *)user_data;
    static PolyPointList P;
    static VertexNumList V;
    static EqList E;
    static WS5 local_ws[MAX_LOCAL_WS];
    static long row_id = 0;

    int i, n_local;

    row_id++;
    S->rows_seen++;

    if (S->rows_seen > S->stop_after_rows) {
        return 1; /* clean stop */
    }

    if (row->vertex_count >= 0 && row->vertex_count < 65) {
        S->vc_hist[row->vertex_count]++;
    }

    if (row->dim != TARGET_DIM) return 0;

    if (S->vertex_count_filter > 0 && row->vertex_count != S->vertex_count_filter) {
        return 0;
    }

    memset(&P, 0, sizeof(P));
    memset(&V, 0, sizeof(V));
    memset(&E, 0, sizeof(E));

    if (!row_to_poly(row, &P, &V)) return 0;
    if (!sanitize_poly_vertices(&P)) return 0;

    if (!Ref_Check(&P, &V, &E)) return 0;
    S->rows_reflexive++;

    n_local = extract_ws5_from_dual_vertices(&P, &V, &E, local_ws, MAX_LOCAL_WS);

    for (i = 0; i < n_local; i++) {
        write_ws5(S->fout, row_id, &local_ws[i]);
        S->ws_written++;
    }

    if ((S->rows_seen % 1000) == 0) {
        fprintf(stderr,
                "[stats] seen=%ld reflexive=%ld ws_written=%ld\n",
                S->rows_seen, S->rows_reflexive, S->ws_written);
    }

    return 0;
}

/* ---------- main ---------- */

int main(int argc, char **argv) {
    Stats S;
    int i;

    memset(&S, 0, sizeof(S));
    inFILE = stdin;
    outFILE = stdout;

    if (argc < 3) {
        fprintf(stderr,
                "Usage: %s <dataset_dir> <ws5_output.txt> [max_rows] [vertex_count_filter]\n",
                argv[0]);
        return 1;
    }

    S.stop_after_rows = MAX_SCAN_ROWS;
    if (argc >= 4) {
        S.stop_after_rows = atol(argv[3]);
        if (S.stop_after_rows <= 0) S.stop_after_rows = MAX_SCAN_ROWS;
    }

    S.vertex_count_filter = 0;
    if (argc >= 5) {
        S.vertex_count_filter = atoi(argv[4]);
        if (S.vertex_count_filter < 0) S.vertex_count_filter = 0;
    }

    S.fout = fopen(argv[2], "w");
    if (!S.fout) {
        perror("fopen output");
        return 2;
    }

    fprintf(stderr, "Scanning dataset dir: %s\n", argv[1]);
    fprintf(stderr, "Writing WS5 to: %s\n", argv[2]);
    fprintf(stderr, "Subset limit: %ld rows\n", S.stop_after_rows);
    fprintf(stderr, "vertex_count filter: %d (0 means no filter)\n", S.vertex_count_filter);

    if (ks_scan_arrow_dir(argv[1], process_row, &S) != 0) {
        fprintf(stderr, "Arrow scan failed\n");
        fclose(S.fout);
        return 3;
    }

    fclose(S.fout);

    fprintf(stderr,
            "done: seen=%ld reflexive=%ld ws_written=%ld\n",
            S.rows_seen, S.rows_reflexive, S.ws_written);

    fprintf(stderr, "vertex_count histogram:\n");
    for (i = 0; i < 65; i++) {
        if (S.vc_hist[i] > 0) {
            fprintf(stderr, "  vc=%d -> %ld\n", i, S.vc_hist[i]);
        }
    }

    return 0;
}