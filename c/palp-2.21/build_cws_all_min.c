#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Global.h"
#include "Subpoly.h"

FILE *inFILE = NULL;
FILE *outFILE = NULL;

/* from Coord.c */
void Make_CWS_Points(CWS *_C, PolyPointList *_P);

#define TARGET_WLEN 5
#define MAX_WS      1000000

typedef struct {
    long row_id;
    Long w[TARGET_WLEN];
} WS5Rec;

static WS5Rec *g_ws = NULL;
static long g_n_ws = 0;

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

static void normalize_cws_row(Long *w, int n) {
    Long common = 0;
    for (int i = 0; i < n; i++) {
        if (w[i] > 0) {
            if (common == 0) common = w[i];
            else common = gcd_long(common, w[i]);
        }
    }
    if (common > 1) {
        for (int i = 0; i < n; i++) w[i] /= common;
    }
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

static int tuple_equal(const Long *a, const Long *b, int n) {
    int i;
    for (i = 0; i < n; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

static int cmp_ws5(const void *A, const void *B) {
    const WS5Rec *a = (const WS5Rec *)A;
    const WS5Rec *b = (const WS5Rec *)B;
    int i;

    for (i = 0; i < TARGET_WLEN; i++) {
        if (a->w[i] < b->w[i]) return -1;
        if (a->w[i] > b->w[i]) return 1;
    }
    if (a->row_id < b->row_id) return -1;
    if (a->row_id > b->row_id) return 1;
    return 0;
}

static int same_ws5(const WS5Rec *a, const WS5Rec *b) {
    return tuple_equal(a->w, b->w, TARGET_WLEN);
}

static int popcount5(int mask) {
    int c = 0;
    while (mask) {
        c += (mask & 1);
        mask >>= 1;
    }
    return c;
}

static void subset_from_mask(const Long src[5], int mask, Long *dst, int *len) {
    int i, k = 0;
    for (i = 0; i < 5; i++) {
        if (mask & (1 << i)) dst[k++] = src[i];
    }
    *len = k;
}

static void normalize_cws_rows(CWS *C) {
    int r, i;
    for (r = 0; r < C->nw; r++) {
        Long g = 0;
        for (i = 0; i < C->N; i++) g = gcd_long(g, C->W[r][i]);
        g = gcd_long(g, C->d[r]);
        if (g <= 0) g = 1;
        if (g > 1) {
            for (i = 0; i < C->N; i++) C->W[r][i] /= g;
            C->d[r] /= g;
        }
    }
}

static void cws_to_line(const CWS *C, char *buf, size_t buflen) {
    int r, i;
    int pos = 0;

    for (r = 0; r < C->nw; r++) {
        pos += snprintf(buf + pos, buflen - pos, "%ld ", C->d[r]);
        for (i = 0; i < C->N; i++) {
            pos += snprintf(buf + pos, buflen - pos, "%ld ", C->W[r][i]);
        }
    }
    snprintf(buf + pos, buflen - pos, "\n");
}

/*
  Find a common subset of exact size c between two 5-tuples, WITHOUT
  renormalizing sub-blocks independently.

  Output:
    left_only  = complement in A   (length 5-c)
    common     = shared block      (length c)
    right_only = complement in B   (length 5-c)
*/
static int split_overlap_exact_size(const Long a[5], const Long b[5], int c,
                                    Long *left_only, Long *common, Long *right_only)
{
    int ma, mb;

    for (ma = 0; ma < (1 << 5); ma++) {
        if (popcount5(ma) != c) continue;

        for (mb = 0; mb < (1 << 5); mb++) {
            Long sa[5], sb[5], ca[5], cb[5];
            int lsa, lsb, lca, lcb;

            if (popcount5(mb) != c) continue;

            subset_from_mask(a, ma, sa, &lsa);
            subset_from_mask(b, mb, sb, &lsb);

            sort_long_array(sa, lsa);
            sort_long_array(sb, lsb);

            if (!tuple_equal(sa, sb, c)) continue;

            subset_from_mask(a, ((1 << 5) - 1) ^ ma, ca, &lca);
            subset_from_mask(b, ((1 << 5) - 1) ^ mb, cb, &lcb);

            if (lca != 5 - c || lcb != 5 - c) continue;

            sort_long_array(ca, lca);
            sort_long_array(cb, lcb);

            memcpy(common, sa, c * sizeof(Long));
            memcpy(left_only, ca, lca * sizeof(Long));
            memcpy(right_only, cb, lcb * sizeof(Long));

            return 1;
        }
    }

    return 0;
}

/*
  Build a generic 2-row CWS from two 5-weight systems sharing an exact
  common subset of size c.

  Total columns = (5-c) + c + (5-c) = 10-c

  row 0 = [left_only | common | 0]
  row 1 = [0 | common | right_only]
*/
static int build_cws_pair_generic(const WS5Rec *a, const WS5Rec *b, int c, CWS *C) {
    Long left_only[5], common[5], right_only[5];
    int i, r; // 'r' deklariert
    int left_len = 5 - c;
    int right_len = 5 - c;
    int ncols = 10 - c;

    if (c < 1 || c > 4) return 0;
    if (ncols > 7) return 0; // PALP Limit Check

    if (!split_overlap_exact_size(a->w, b->w, c, left_only, common, right_only)) {
        return 0;
    }

    memset(C, 0, sizeof(CWS));
    C->nw = 2;
    C->N = ncols;
    C->index = 1;

    /* row 0 */
    for (i = 0; i < left_len; i++) C->W[0][i] = left_only[i];
    for (i = 0; i < c; i++) C->W[0][left_len + i] = common[i];

    /* row 1 */
    for (i = 0; i < c; i++) C->W[1][left_len + i] = common[i];
    for (i = 0; i < right_len; i++) C->W[1][left_len + c + i] = right_only[i];

    // GGT-Normalisierung gegen den VZ_to_Base Fehler
    normalize_cws_row(C->W[0], 7);
    normalize_cws_row(C->W[1], 7);

    // Finale Summenberechnung
    for (r = 0; r < 2; r++) {
        C->d[r] = 0;
        for (i = 0; i < 7; i++) C->d[r] += C->W[r][i];
    }
    
    return 1;
}

static int cws_is_reflexive_and_minimal_internal(const CWS *Cin) {
    static PolyPointList P;
    static VertexNumList V;
    static EqList E;
    CWS C;

    if (Cin->N - Cin->nw != 5) return 0;

    memset(&P, 0, sizeof(P));
    memset(&V, 0, sizeof(V));
    memset(&E, 0, sizeof(E));
    memcpy(&C, Cin, sizeof(CWS));

    Make_CWS_Points(&C, &P);
    
    if (!Ref_Check(&P, &V, &E)) return 0;
    if (!Poly_Min_check(&P, &V, &E)) return 0;

    return 1;
}

static void write_cws(FILE *fout, const WS5Rec *a, const WS5Rec *b, int c, const CWS *C) {
    char line[2048];

    fprintf(fout, "# overlap=%d rows=%ld,%ld\n", c, a->row_id, b->row_id);
    cws_to_line(C, line, sizeof(line));
    fputs(line, fout);
}

static long load_ws5_file(const char *filename) {
    FILE *f;
    long cap = 100000;
    long row_id;
    Long w1, w2, w3, w4, w5;

    g_ws = (WS5Rec *)malloc(cap * sizeof(WS5Rec));
    if (!g_ws) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }

    f = fopen(filename, "r");
    if (!f) {
        perror("fopen ws5 input");
        exit(1);
    }

    while (fscanf(f, "%ld %ld %ld %ld %ld %ld",
                  &row_id, &w1, &w2, &w3, &w4, &w5) == 6) {
        if (g_n_ws >= cap) {
            cap *= 2;
            g_ws = (WS5Rec *)realloc(g_ws, cap * sizeof(WS5Rec));
            if (!g_ws) {
                fprintf(stderr, "realloc failed\n");
                exit(1);
            }
        }

        g_ws[g_n_ws].row_id = row_id;
        g_ws[g_n_ws].w[0] = w1;
        g_ws[g_n_ws].w[1] = w2;
        g_ws[g_n_ws].w[2] = w3;
        g_ws[g_n_ws].w[3] = w4;
        g_ws[g_n_ws].w[4] = w5;
        sort_long_array(g_ws[g_n_ws].w, 5);
        g_n_ws++;
    }

    fclose(f);
    return g_n_ws;
}

static long dedup_ws5(void) {
    long i, out;

    if (g_n_ws == 0) return 0;

    qsort(g_ws, g_n_ws, sizeof(WS5Rec), cmp_ws5);

    out = 1;
    for (i = 1; i < g_n_ws; i++) {
        if (!same_ws5(&g_ws[i], &g_ws[out - 1])) {
            if (out != i) g_ws[out] = g_ws[i];
            out++;
        }
    }

    g_n_ws = out;
    return g_n_ws;
}

int main(int argc, char **argv) {
    FILE *fout;
    long i, j;
    long n_raw, n_unique;
    long n_candidates = 0;
    long n_hits = 0;
    long stop_after_candidates = 0;
    int c;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <ws5.txt> <cws_out.txt> [max_candidates]\n", argv[0]);
        return 1;
    }

    if (argc >= 4) {
        stop_after_candidates = atol(argv[3]);
        if (stop_after_candidates < 0) stop_after_candidates = 0;
    }

    fout = fopen(argv[2], "w");
    if (!fout) {
        perror("fopen output");
        return 2;
    }

    n_raw = load_ws5_file(argv[1]);
    fprintf(stderr, "Loaded WS5 records: %ld\n", n_raw);

    n_unique = dedup_ws5();
    fprintf(stderr, "Unique WS5 records: %ld\n", n_unique);

    for (i = 0; i < g_n_ws; i++) {
        for (j = i + 1; j < g_n_ws; j++) {
            for (c = 3; c <= 3; c++) {
                CWS C;

                if (stop_after_candidates > 0 && n_candidates >= stop_after_candidates) break;

                if (!build_cws_pair_generic(&g_ws[i], &g_ws[j], c, &C)) continue;

                n_candidates++;

                if (n_candidates <= 10) {
                    char line[2048];
                    cws_to_line(&C, line, sizeof(line));
                    fprintf(stderr, "[debug] candidate %ld: %s", n_candidates, line);
                }

                if (cws_is_reflexive_and_minimal_internal(&C)) {
                    write_cws(fout, &g_ws[i], &g_ws[j], c, &C);
                    fflush(fout);
                    n_hits++;
                }

                if ((n_candidates % 1000) == 0) {
                    fprintf(stderr, "[stats] candidates=%ld hits=%ld\n", n_candidates, n_hits);
                }
            }

            if (stop_after_candidates > 0 && n_candidates >= stop_after_candidates) break;
        }

        if (stop_after_candidates > 0 && n_candidates >= stop_after_candidates) break;
    }

    fclose(fout);
    free(g_ws);

    fprintf(stderr, "Done. candidates=%ld minimal_reflexive_hits=%ld\n", n_candidates, n_hits);
    return 0;
}