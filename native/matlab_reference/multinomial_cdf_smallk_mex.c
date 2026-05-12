/* multinomial_cdf_smallk_mex.c
 *
 * H = multinomial_cdf_smallk_mex(p, N, m, seed)
 *   p    : 1-by-d (or d-by-1) double probabilities (nonnegative)
 *   N    : number of trials per row (size_t)
 *   m    : number of rows (size_t)
 *   seed : uint64 or double (optional; default 7)
 *
 * Output:
 *   H : m-by-d int32 histogram; each row sums to N
 *
 * Notes:
 * - Deterministic across thread counts (per-row splitmix64 substreams).
 * - Uses binary search (lower_bound) on CDF for categorical draws.
 */

#include "mex.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

/* ---------------- RNG: splitmix64 + (0,1) double ---------------- */
static inline uint64_t splitmix64(uint64_t *s){
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline double u01(uint64_t *s){                 /* (0,1) open interval */
    const uint64_t r = splitmix64(s);
    const uint64_t m = (r >> 11) | 1ULL;               /* avoid exact 0 */
    return (double)m * (1.0/9007199254740992.0);       /* 2^53 */
}

/* ---------------- helpers ---------------- */
static inline uint64_t read_seed_any(const mxArray *a, uint64_t def){
    if (!a) return def;
    if (mxIsUint64(a)){
        const uint64_t *sp = (const uint64_t*)mxGetData(a);
        return sp ? *sp : def;
    } else {
        double v = mxGetScalar(a);
        if (!(v==v)) return def;
        return (uint64_t) llround(v);
    }
}

static inline size_t lower_bound_cdf(const double *cdf, size_t d, double u){
    size_t lo = 0, hi = d - 1;
    while (lo < hi){
        size_t mid = lo + ((hi - lo) >> 1);
        if (cdf[mid] >= u) hi = mid; else lo = mid + 1;
    }
    return lo; /* first index with cdf[idx] >= u */
}

/* ---------------- mex entry ---------------- */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 3) {
        mexErrMsgIdAndTxt("multinom_cdf_smallk:args",
            "Need p, N, m [, seed]");
    }

    /* p */
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("multinom_cdf_smallk:p","p must be real double.");
    }
    const double *p_in = mxGetPr(prhs[0]);
    const size_t d = (size_t) mxGetNumberOfElements(prhs[0]);
    if (d < 1) mexErrMsgIdAndTxt("multinom_cdf_smallk:p","p must be non-empty.");

    /* N, m */
    const double Nd = mxGetScalar(prhs[1]);
    const double md = mxGetScalar(prhs[2]);
    if (!(Nd >= 0) || floor(Nd) != Nd) mexErrMsgIdAndTxt("multinom_cdf_smallk:N","N must be a nonnegative integer.");
    if (!(md >= 0) || floor(md) != md) mexErrMsgIdAndTxt("multinom_cdf_smallk:m","m must be a nonnegative integer.");
    const size_t N = (size_t) Nd;
    const size_t m = (size_t) md;

    /* seed (optional) */
    uint64_t seed = 7ULL;
    if (nrhs >= 4) seed = read_seed_any(prhs[3], 7ULL);

    /* normalize p and build CDF */
    double *cdf = (double*) mxCalloc(d, sizeof(double));
    double sum = 0.0;
    for (size_t j=0; j<d; ++j){
        double pj = p_in[j];
        if (!(pj >= 0.0) || !mxIsFinite(pj)){
            mxFree(cdf);
            mexErrMsgIdAndTxt("multinom_cdf_smallk:p","p must be nonnegative and finite.");
        }
        sum += pj;
    }
    if (!(sum > 0.0)){
        mxFree(cdf);
        mexErrMsgIdAndTxt("multinom_cdf_smallk:p","Sum(p) must be positive.");
    }
    double acc = 0.0;
    for (size_t j=0; j<d; ++j){
        acc += p_in[j] / sum;
        cdf[j] = acc;
    }
    cdf[d-1] = 1.0;  /* force last entry to 1 exactly */

    /* output m-by-d int32 */
    plhs[0] = mxCreateNumericMatrix((mwSize)m, (mwSize)d, mxINT32_CLASS, mxREAL);
    int32_t *out = (int32_t*) mxGetData(plhs[0]);

    /* handle N==0 quickly */
    if (N == 0){
        /* already zero-initialized */
        mxFree(cdf);
        return;
    }

    /* Parallel over rows; independent per-row substreams derived from seed */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
    #endif
    for (ptrdiff_t i = 0; i < (ptrdiff_t)m; ++i){
        uint64_t s = seed ^ (0xD1B54A32D192ED03ULL * ((uint64_t)i + 1ULL));

        /* local counts for this row */
        int32_t *row_counts = (int32_t*) mxCalloc(d, sizeof(int32_t));
        for (size_t t=0; t<N; ++t){
            double u = u01(&s);
            size_t k = lower_bound_cdf(cdf, d, u);
            row_counts[k] += 1;
        }

        /* write row i into column-major output: out[i + j*m] */
        for (size_t j=0; j<d; ++j){
            out[i + j*(size_t)m] = row_counts[j];
        }
        mxFree(row_counts);
    }

    mxFree(cdf);
}

