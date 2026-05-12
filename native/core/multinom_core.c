#include "multinom_core.h"
#include "binom_core.h"
#include "rng_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
  #include <omp.h>
  #define CS_OMP_PRAGMA(x) _Pragma(#x)
#else
  #define CS_OMP_PRAGMA(x)
#endif

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

static size_t lower_bound_cdf_(const double *cdf, size_t d, double u) {
    size_t lo = 0;
    size_t hi = d - 1;
    while (lo < hi) {
        size_t mid = lo + ((hi - lo) >> 1);
        if (u <= cdf[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

static size_t alias_sample_one_(const cs_multinom_alias_state *state, cs_rng64_t *rng) {
    double u = cs_rng64_u01_open(rng);
    size_t k = (size_t)((double)state->d * u);
    if (k >= state->d) {
        k = state->d - 1;
    }
    return (cs_rng64_u01_open(rng) < state->prob[k]) ? k : state->alias[k];
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
    CS_OMP_PRAGMA(omp parallel for schedule(static) if(m > 1))
    for (ptrdiff_t row_i = 0; row_i < (ptrdiff_t)m; ++row_i) {
        size_t row = (size_t)row_i;
        uint64_t row_seed = seed ^ (UINT64_C(0x60642E2A34326F15) * (uint64_t)(row + 1));
        cs_multinom_pivot_draw_one(state, K, row_seed, out_row_major + row * d);
    }
}

int cs_multinom_cdf_build(const double *p, size_t d, cs_multinom_cdf_state *state) {
    if (state == NULL) {
        return CS_MULTINOM_BAD_PROB;
    }
    state->d = 0;
    state->cdf = NULL;
    if (p == NULL || d == 0) {
        return CS_MULTINOM_BAD_PROB;
    }

    double total = 0.0;
    for (size_t i = 0; i < d; ++i) {
        if (!isfinite(p[i]) || p[i] < 0.0) {
            return CS_MULTINOM_BAD_PROB;
        }
        total += p[i];
    }
    if (!(total > 0.0)) {
        return CS_MULTINOM_BAD_PROB;
    }

    double *cdf = (double *)malloc(d * sizeof(double));
    if (cdf == NULL) {
        return CS_MULTINOM_ALLOC_FAILED;
    }
    double acc = 0.0;
    for (size_t i = 0; i < d; ++i) {
        acc += p[i] / total;
        cdf[i] = acc;
    }
    cdf[d - 1] = 1.0;
    state->d = d;
    state->cdf = cdf;
    return CS_MULTINOM_OK;
}

void cs_multinom_cdf_free(cs_multinom_cdf_state *state) {
    if (state == NULL) {
        return;
    }
    free(state->cdf);
    state->d = 0;
    state->cdf = NULL;
}

void cs_multinom_cdf_draw_batch(const cs_multinom_cdf_state *state,
                                uint64_t K,
                                size_t m,
                                uint64_t seed,
                                int64_t *out_row_major) {
    const size_t d = state->d;
    CS_OMP_PRAGMA(omp parallel for schedule(static) if(m > 1))
    for (ptrdiff_t row_i = 0; row_i < (ptrdiff_t)m; ++row_i) {
        size_t row = (size_t)row_i;
        int64_t *out = out_row_major + row * d;
        memset(out, 0, d * sizeof(int64_t));
        uint64_t row_seed = seed ^ (UINT64_C(0xD1B54A32D192ED03) * (uint64_t)(row + 1));
        cs_rng64_t rng;
        cs_rng64_seed(&rng, row_seed);
        for (uint64_t t = 0; t < K; ++t) {
            size_t k = lower_bound_cdf_(state->cdf, d, cs_rng64_u01_open(&rng));
            out[k] += 1;
        }
    }
}

int cs_multinom_alias_build(const double *p, size_t d, cs_multinom_alias_state *state) {
    if (state == NULL) {
        return CS_MULTINOM_BAD_PROB;
    }
    state->d = 0;
    state->prob = NULL;
    state->alias = NULL;
    if (p == NULL || d == 0) {
        return CS_MULTINOM_BAD_PROB;
    }

    double total = 0.0;
    for (size_t i = 0; i < d; ++i) {
        if (!isfinite(p[i]) || p[i] < 0.0) {
            return CS_MULTINOM_BAD_PROB;
        }
        total += p[i];
    }
    if (!(total > 0.0)) {
        return CS_MULTINOM_BAD_PROB;
    }

    double *prob = (double *)malloc(d * sizeof(double));
    size_t *alias = (size_t *)malloc(d * sizeof(size_t));
    size_t *small = (size_t *)malloc(d * sizeof(size_t));
    size_t *large = (size_t *)malloc(d * sizeof(size_t));
    if (prob == NULL || alias == NULL || small == NULL || large == NULL) {
        free(prob);
        free(alias);
        free(small);
        free(large);
        return CS_MULTINOM_ALLOC_FAILED;
    }

    size_t nsmall = 0;
    size_t nlarge = 0;
    for (size_t i = 0; i < d; ++i) {
        prob[i] = (p[i] / total) * (double)d;
        if (prob[i] < 1.0) {
            small[nsmall++] = i;
        } else {
            large[nlarge++] = i;
        }
    }

    while (nsmall && nlarge) {
        size_t s = small[--nsmall];
        size_t l = large[--nlarge];
        alias[s] = l;
        prob[l] = (prob[l] + prob[s]) - 1.0;
        if (prob[l] < 1.0) {
            small[nsmall++] = l;
        } else {
            large[nlarge++] = l;
        }
    }
    while (nlarge) {
        size_t l = large[--nlarge];
        prob[l] = 1.0;
        alias[l] = l;
    }
    while (nsmall) {
        size_t s = small[--nsmall];
        prob[s] = 1.0;
        alias[s] = s;
    }

    free(small);
    free(large);
    state->d = d;
    state->prob = prob;
    state->alias = alias;
    return CS_MULTINOM_OK;
}

void cs_multinom_alias_free(cs_multinom_alias_state *state) {
    if (state == NULL) {
        return;
    }
    free(state->prob);
    free(state->alias);
    state->d = 0;
    state->prob = NULL;
    state->alias = NULL;
}

void cs_multinom_alias_draw_batch(const cs_multinom_alias_state *state,
                                  uint64_t K,
                                  size_t m,
                                  uint64_t seed,
                                  int64_t *out_row_major) {
    const size_t d = state->d;
    CS_OMP_PRAGMA(omp parallel for schedule(static) if(m > 1))
    for (ptrdiff_t row_i = 0; row_i < (ptrdiff_t)m; ++row_i) {
        size_t row = (size_t)row_i;
        int64_t *out = out_row_major + row * d;
        memset(out, 0, d * sizeof(int64_t));
        uint64_t row_seed = seed ^ (UINT64_C(0xD1B54A32D192ED03) * (uint64_t)(row + 1));
        cs_rng64_t rng;
        cs_rng64_seed(&rng, row_seed);
        for (uint64_t t = 0; t < K; ++t) {
            size_t k = alias_sample_one_(state, &rng);
            out[k] += 1;
        }
    }
}
