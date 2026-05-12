
/* multinomial_poisson_pivot_hist_mex.c  (pivot-only; exact)
 *
 * Usage:
 *   X = multinomial_poisson_pivot_hist_mex(p, N, m [, seed])
 *
 * Inputs:
 *   p    : double vector of length d (K categories), any shape (Kx1 or 1xK)
 *   N    : number of trials per draw (nonnegative scalar integer)
 *   m    : number of independent draws (nonnegative scalar integer)
 *   seed : uint64 or double (optional; default 0xABCDEF9876543210)
 *
 * Output:
 *   X : (m x d) int32 matrix; EACH ROW sums to N.
 *
 * Kernel:
 *   - Pivoted sequential binomial:
 *       * Sort p descending once → p_sorted, perm
 *       * For each row j:
 *           for r = 1..d-1:
 *             x_r ~ Bin(N_rem, p_sorted[r]/p_rem)
 *             write X(j, perm[r]) = x_r
 *             update N_rem, p_rem
 *           X(j, perm[d]) = N_rem  // remainder
 *
 * Determinism:
 *   - Per-row RNG stream: base = seed ^ C*(j+1) → reproducible across threads.
 *
 * Build (example):
 *   mex -O CFLAGS="\$CFLAGS -O3 -fopenmp -march=native -mtune=native" \
 *       LDFLAGS="\$LDFLAGS -fopenmp" \
 *       multinomial_poisson_pivot_hist_mex.c multinomial_core.c
 *
 * Notes:
 *   - Requires sampler_core.h to provide rng64_t, rng64_seed, and
 *     sample_binom_center_inversion(rng64_t*, int n, double p).
 *   - N must fit in a 32-bit int for the binomial inner call.
 */

#include "mex.h"
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "sampler_core.h"  /* rng64_t, rng64_seed, sample_binom_center_inversion */

#ifndef OMP_CHUNK
#define OMP_CHUNK 1024
#endif

/* ---- helpers ---- */
static int64_t read_int64_scalar(const mxArray *a, const char *name){
    if (mxIsInt64(a)) {
        const int64_t *p = (const int64_t*)mxGetData(a);
        if (!p) mexErrMsgIdAndTxt("multinom:scalar", "%s missing data", name);
        return *p;
    } else if (mxIsInt32(a)) {
        const int32_t *p = (const int32_t*)mxGetData(a);
        if (!p) mexErrMsgIdAndTxt("multinom:scalar", "%s missing data", name);
        return (int64_t)(*p);
    } else if (mxIsDouble(a)) {
        double v = mxGetScalar(a);
        if (!(v==v) || !mxIsDouble(a)) mexErrMsgIdAndTxt("multinom:scalar", "%s invalid", name);
        return (int64_t) llround(v);
    } else {
        mexErrMsgIdAndTxt("multinom:scalar", "%s must be a numeric scalar", name);
        return 0;
    }
}

typedef struct { double val; int idx; } pair_t;
static int cmp_desc(const void *a, const void *b){
    const pair_t *x=(const pair_t*)a, *y=(const pair_t*)b;
    return (x->val < y->val) ? 1 : (x->val > y->val ? -1 : 0);
}

/* ---- mex entry ---- */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 3 || nrhs > 4)
        mexErrMsgIdAndTxt("multinom:args", "Need p, N, m [, seed]");

    /* p: double vector length d */
    if (!mxIsDouble(prhs[0]))
        mexErrMsgIdAndTxt("multinom:p", "p must be double");
    const mwSize d_m = mxGetM(prhs[0]), d_n = mxGetN(prhs[0]);
    int d = (int)(d_m * d_n);
    if (d < 1)
        mexErrMsgIdAndTxt("multinom:p", "p must be non-empty");
    const double *p_in = mxGetPr(prhs[0]);

    /* N: trials per draw (nonnegative) */
    int64_t N64 = read_int64_scalar(prhs[1], "N");
    if (N64 < 0) mexErrMsgIdAndTxt("multinom:N", "N must be >= 0");
    if (N64 > INT32_MAX)
        mexErrMsgIdAndTxt("multinom:N", "N too large for current kernel (max 2^31-1)");
    int64_t N_input = N64;   /* keep as 64-bit for loop arithmetic */

    /* m: number of draws (nonnegative) */
    int64_t m = read_int64_scalar(prhs[2], "m");
    if (m < 0) mexErrMsgIdAndTxt("multinom:m", "m must be >= 0");

    /* seed (optional) */
    uint64_t seed = 0xABCDEF9876543210ULL;
    if (nrhs >= 4) {
        if (mxIsUint64(prhs[3])) {
            const uint64_t *sp = (const uint64_t*)mxGetData(prhs[3]);
            if (sp) seed = *sp;
        } else if (mxIsDouble(prhs[3])) {
            seed = (uint64_t) llround(mxGetScalar(prhs[3]));
        } /* else ignore */
    }

    /* Output: m x d int32, COLUMN-MAJOR (MATLAB) */
    plhs[0] = mxCreateNumericMatrix((mwSize)m, (mwSize)d, mxINT32_CLASS, mxREAL);
    int32_t *X = (int32_t*)mxGetData(plhs[0]);

    /* Normalize p */
    double *pnorm = (double*)mxCalloc((mwSize)d, sizeof(double));
    double sum = 0.0;
    for (int i=0;i<d;i++) sum += p_in[i];
    if (!(sum > 0.0)) {
        mxFree(pnorm);
        mexErrMsgIdAndTxt("multinom:p", "Sum(p) must be positive");
    }
    for (int i=0;i<d;i++) pnorm[i] = p_in[i] / sum;

    /* Sort p descending once; keep permutation */
    pair_t *pairs = (pair_t*)mxCalloc((mwSize)d, sizeof(pair_t));
    for (int i=0;i<d;i++){ pairs[i].val = pnorm[i]; pairs[i].idx = i; }
    qsort(pairs, (size_t)d, sizeof(pair_t), cmp_desc);

    double *p_sorted = (double*)mxCalloc((mwSize)d, sizeof(double));
    int    *perm     = (int*)   mxCalloc((mwSize)d, sizeof(int));
    for (int i=0;i<d;i++){ p_sorted[i] = pairs[i].val; perm[i] = pairs[i].idx; }
    mxFree(pairs);

    /* Parallel over rows (draws). Deterministic via per-row seed. */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static, OMP_CHUNK)
    #endif
    for (int64_t j=0; j<m; ++j) {
        /* each row gets its own stream; independent of scheduling */
        rng64_t rr;
        uint64_t base = seed ^ (0x60642E2A34326F15ULL * (uint64_t)(j + 1));
        rng64_seed(&rr, base);

        /* Optionally zero row (not strictly required, but safe) */
        for (int c=0; c<d; ++c) {
            X[(size_t)j + (size_t)m * (size_t)c] = 0;
        }

        int64_t Nrem = N_input;
        double  prem = 1.0;

        /* sequential binomials on sorted probs, except last category */
        for (int r=0; r<d-1; ++r) {
            if (Nrem <= 0) break;
            double pr = (prem <= 0.0) ? 0.0 : (p_sorted[r] / prem);
            if (pr < 0.0) pr = 0.0;
            if (pr > 1.0) pr = 1.0;

            int x = sample_binom_center_inversion(&rr, (int)Nrem, pr);

            /* write X(j, perm[r]) in column-major: idx = j + m * col */
            X[(size_t)j + (size_t)m * (size_t)perm[r]] = (int32_t)x;

            Nrem -= (int64_t)x;
            prem -= p_sorted[r];
        }

        /* remainder goes to the last category */
        if (Nrem > 0) {
            X[(size_t)j + (size_t)m * (size_t)perm[d-1]] = (int32_t)Nrem;
        }
    }

    mxFree(p_sorted);
    mxFree(perm);
    mxFree(pnorm);
}
