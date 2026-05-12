#include "multinom_core.h"
#include "binom_core.h"
#include "rng_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double value;
    size_t index;
} prob_pair_t;

static int compare_prob_desc_(const void *a, const void *b) {
    const prob_pair_t *pa = (const prob_pair_t *)a;
    const prob_pair_t *pb = (const prob_pair_t *)b;
    if (pa->value < pb->value) {
        return 1;
    }
    if (pa->value > pb->value) {
        return -1;
    }
    return 0;
}

static double rng_u01_cb_(void *ctx) {
    return cs_rng64_u01_open((cs_rng64_t *)ctx);
}

int cs_multinom_pivot_build(const double *p, size_t d, cs_multinom_pivot_state *state) {
    if (state == NULL) {
        return CS_MULTINOM_BAD_PROB;
    }
    state->d = 0;
    state->p_sorted = NULL;
    state->perm = NULL;

    if (p == NULL || d == 0) {
        return CS_MULTINOM_BAD_PROB;
    }

    double total = 0.0;
    for (size_t i = 0; i < d; ++i) {
        const double pi = p[i];
        if (!isfinite(pi) || pi < 0.0) {
            return CS_MULTINOM_BAD_PROB;
        }
        total += pi;
    }
    if (!(total > 0.0)) {
        return CS_MULTINOM_BAD_PROB;
    }

    prob_pair_t *pairs = (prob_pair_t *)malloc(d * sizeof(prob_pair_t));
    double *p_sorted = (double *)malloc(d * sizeof(double));
    size_t *perm = (size_t *)malloc(d * sizeof(size_t));
    if (pairs == NULL || p_sorted == NULL || perm == NULL) {
        free(pairs);
        free(p_sorted);
        free(perm);
        return CS_MULTINOM_ALLOC_FAILED;
    }

    for (size_t i = 0; i < d; ++i) {
        pairs[i].value = p[i] / total;
        pairs[i].index = i;
    }
    qsort(pairs, d, sizeof(prob_pair_t), compare_prob_desc_);

    for (size_t i = 0; i < d; ++i) {
        p_sorted[i] = pairs[i].value;
        perm[i] = pairs[i].index;
    }
    free(pairs);

    state->d = d;
    state->p_sorted = p_sorted;
    state->perm = perm;
    return CS_MULTINOM_OK;
}

void cs_multinom_pivot_free(cs_multinom_pivot_state *state) {
    if (state == NULL) {
        return;
    }
    free(state->p_sorted);
    free(state->perm);
    state->d = 0;
    state->p_sorted = NULL;
    state->perm = NULL;
}

void cs_multinom_pivot_draw_one(const cs_multinom_pivot_state *state,
                                uint64_t K,
                                uint64_t seed,
                                int64_t *out) {
    const size_t d = state->d;
    memset(out, 0, d * sizeof(int64_t));
    if (d == 0) {
        return;
    }
    if (K == 0) {
        return;
    }

    cs_rng64_t rng;
    cs_rng64_seed(&rng, seed);
    size_t rem = (size_t)K;
    double prem = 1.0;

    for (size_t r = 0; r + 1 < d; ++r) {
        if (rem == 0) {
            break;
        }
        double pr = (prem <= 0.0) ? 0.0 : state->p_sorted[r] / prem;
        if (pr < 0.0) {
            pr = 0.0;
        } else if (pr > 1.0) {
            pr = 1.0;
        }
        size_t x = binom_centerout_core(rem, pr, rng_u01_cb_, &rng);
        out[state->perm[r]] = (int64_t)x;
        rem -= x;
        prem -= state->p_sorted[r];
    }
    out[state->perm[d - 1]] = (int64_t)rem;
}

void cs_multinom_pivot_draw_batch(const cs_multinom_pivot_state *state,
                                  uint64_t K,
                                  size_t m,
                                  uint64_t seed,
                                  int64_t *out_row_major) {
    const size_t d = state->d;
    for (size_t row = 0; row < m; ++row) {
        uint64_t row_seed = seed ^ (UINT64_C(0x60642E2A34326F15) * (uint64_t)(row + 1));
        cs_multinom_pivot_draw_one(state, K, row_seed, out_row_major + row * d);
    }
}
