// ================================================================
// multinomial_core.c  (core helpers used by multinomial_kernels_mex)
// - Alias (Vose) build/sample
// - Small-K CDF sampler for histograms
// - "Thin" alias-based histogram (good for small K, many rows)
// - Pivot histogram, one row and batch (parallel-friendly shell)
// - MATLAB/mx helpers for packing/unpacking state and matrices
//
// NOTE:
//  * All OpenMP pragmas are wrapped via OMP_PRAGMA(...).
//  * The batch pivot uses a robust, deterministic per-row RNG split.
//  * Binomial/Poisson internals are expected to come from binom_core.c
//    (linked at MEX time). We only call its exported functions; no
//    definitions are duplicated here.
// ================================================================

#include "mex.h"
#include "matrix.h"
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

// ---------- OpenMP wrapper ----------
#ifdef _OPENMP
  #include <omp.h>
  #define OMP_PRAGMA(x) _Pragma(#x)
#else
  #define OMP_PRAGMA(x)
#endif

// ---------- tiny utils ----------
static inline double clamp01_open(double u){
    if (!(u>0.0)) u = DBL_MIN;
    if (u>=1.0)   u = 1.0 - DBL_MIN;
    return u;
}
static inline double max2(double a, double b){ return (a>b)?a:b; }

// ================================================================
// RNG: SplitMix64 (deterministic; good for stream-splitting)
// ================================================================
static inline uint64_t splitmix64_next(uint64_t *state){
    uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}
static inline double u01_open(uint64_t *state){
    // 53-bit mantissa to (0,1) open interval
    // (avoid 0 and 1 to be friendly with logs/exps)
    const uint64_t r = splitmix64_next(state);
    // shift to 53 bits
    const uint64_t x = (r >> 11) | UINT64_C(1); // ensure >0
    return (double)x * (1.0/9007199254740992.0); // 2^53
}

// ================================================================
// CDF utilities (for small-K CDF method)
// ================================================================
static inline void scan_prefix(const double *p, mwSize d, double *cdf){
    double s = 0.0;
    for (mwSize i=0;i<d;i++){ s += p[i]; cdf[i] = s; }
    const double Z = s > 0.0 ? s : 1.0;
    for (mwSize i=0;i<d;i++) cdf[i] /= Z;
    // keep strictly < 1 for last
    cdf[d-1] = 1.0 - DBL_MIN;
}
static inline mwSize bsearch_cdf(const double *cdf, mwSize d, double u){
    // u in (0,1)
    mwSize lo=0, hi=d-1;
    while (lo<hi){
        mwSize mid = (lo+hi)>>1;
        if (u <= cdf[mid]) hi = mid; else lo = mid+1;
    }
    return lo;
}

// ================================================================
// Alias method (Vose) build/sample
// ================================================================
typedef struct {
    // raw pointers owned by MATLAB (mxArrays live on the state struct)
    const double  *q;
    const int32_T *J;
    mwSize         d;
} alias_view;

static void alias_build_arrays(const double *p, mwSize d,
                               mxArray **prob_out, mxArray **alias_out)
{
    // Normalize to exactly sum 1
    double sum=0.0; for (mwSize i=0;i<d;i++) sum += p[i];
    if (!(sum>0.0)) mexErrMsgIdAndTxt("multinom:alias","p must sum>0");

    mxArray *prob  = mxCreateDoubleMatrix(1, d, mxREAL);
    mxArray *alias = mxCreateNumericMatrix(1, d, mxINT32_CLASS, mxREAL);

    double  *q = mxGetPr(prob);
    int32_T *J = (int32_T*)mxGetData(alias);

    // Work lists (as int stacks)
    mwSize *small = (mwSize*)mxCalloc(d, sizeof(mwSize));
    mwSize *large = (mwSize*)mxCalloc(d, sizeof(mwSize));
    mwSize ns=0, nl=0;

    const double invd = (double)d / sum;
    for (mwSize i=0;i<d;i++){
        q[i] = p[i] * invd;  // target: mean 1 across bins
        if (q[i] < 1.0) small[ns++] = i;
        else            large[nl++] = i;
    }

    while (ns && nl){
        mwSize s = small[--ns];
        mwSize l = large[--nl];
        J[s] = (int32_T)(l+1);         // 1-based for MATLAB users
        q[l] = (q[l] + q[s]) - 1.0;
        if (q[l] < 1.0) small[ns++] = l;
        else            large[nl++] = l;
    }
    while (nl){ mwSize l = large[--nl]; J[l] = (int32_T)(l+1); }
    while (ns){ mwSize s = small[--ns]; J[s] = (int32_T)(s+1); }

    mxFree(small); mxFree(large);
    *prob_out = prob;
    *alias_out = alias;
}

static mxArray* alias_pack_struct(const mxArray *prob, const mxArray *alias)
{
    const char *fns[] = {"prob","alias","d"};
    mxArray *st = mxCreateStructMatrix(1,1,3,fns);
    mxSetFieldByNumber(st,0,0,(mxArray*)prob);
    mxSetFieldByNumber(st,0,1,(mxArray*)alias);
    mxArray *dval = mxCreateDoubleScalar((double)mxGetNumberOfElements(prob));
    mxSetFieldByNumber(st,0,2,dval);
    return st;
}

static void alias_unpack_struct(const mxArray *st,
                                const double **prob,
                                const int32_T **alias,
                                mwSize *d)
{
    const mxArray *prob_mx  = mxGetField(st,0,"prob");
    const mxArray *alias_mx = mxGetField(st,0,"alias");
    if (!prob_mx || !alias_mx)
        mexErrMsgIdAndTxt("multinom:alias_state","bad alias state struct");
    *prob  = mxGetPr(prob_mx);
    *alias = (const int32_T*)mxGetData(alias_mx);
    *d     = (mwSize)mxGetNumberOfElements(prob_mx);
}

static inline mwSize alias_sample_one(const double *q,
                                      const int32_T *J,
                                      mwSize d,
                                      uint64_t *state)
{
    // Draw column index in 1..d
    double u   = u01_open(state);
    mwSize kk  = (mwSize)(u * (double)d);
    if (kk >= d) kk = d-1;
    double u2  = u01_open(state);
    const double qk = q[kk];
    return (u2 < qk) ? (kk) : ((mwSize)J[kk]-1);
}

static void alias_sample_N_to(const double *q, const int32_T *J,
                              mwSize d, mwSize N, uint64_t seed,
                              int32_T *out)
{
    uint64_t st = seed;
    for (mwSize i=0;i<N;i++){
        out[i] = (int32_T)(alias_sample_one(q,J,d,&st) + 1); // 1..d
    }
}

// ================================================================
// smallK CDF histogram
// ================================================================
static void cdf_smallk_hist(const double *p, mwSize d,
                            uint64_t K, mwSize m, uint64_t seed,
                            double *H /*row-major m-by-d*/)
{
    // Precompute CDF once
    double *cdf = (double*)mxCalloc(d, sizeof(double));
    scan_prefix(p, d, cdf);

    OMP_PRAGMA(omp parallel for schedule(static) if(m>1))
    for (ptrdiff_t r=0; r<(ptrdiff_t)m; ++r){
        uint64_t st = seed + (uint64_t)r * UINT64_C(1442695040888963407);
        double *row = H + (ptrdiff_t)r*(ptrdiff_t)d;
        for (uint64_t t=0;t<K;t++){
            const double u = u01_open(&st);
            const mwSize  k = bsearch_cdf(cdf,d,u);
            row[k] += 1.0;
        }
    }

    mxFree(cdf);
}

// ================================================================
// alias "thin" hist (good when K is small and m is large)
// ================================================================
static void alias_hist_thin(const double *q, const int32_T *J,
                            mwSize d, uint64_t K, mwSize m,
                            uint64_t seed, double *H /*row-major*/)
{
    OMP_PRAGMA(omp parallel for schedule(static) if(m>1))
    for (ptrdiff_t r=0; r<(ptrdiff_t)m; ++r){
        uint64_t st = seed + (uint64_t)r * UINT64_C(1442695040888963407);
        double *row = H + (ptrdiff_t)r*(ptrdiff_t)d;
        for (uint64_t t=0;t<K;t++){
            const mwSize k = alias_sample_one(q,J,d,&st);
            row[k] += 1.0;
        }
    }
}

// ================================================================
// Pivot histogram
//  - We delegate Poisson/Binomial draws to binom_core.*
//  - This shell handles per-row parallelism and exact sum-to-K fix.
// ================================================================
/* binom_core exports (linked at MEX link step)
 *   We only declare what we need here. If your binom_core.h already
 *   exists on your include path, you can #include it instead.
 */
#ifndef BINOM_CORE_DECLS
#define BINOM_CORE_DECLS
// Draw Poisson(lambda) with caller-supplied U(0,1) and U64 streams.
// Returns int64 (>=0).
extern int64_t binomcore_poisson_draw(double lambda,
                                      double (*u01)(uint64_t*),
                                      uint64_t *state);
// Draw Binomial(n,p) (n up to 2e9) with same RNG callbacks.
extern int64_t binomcore_binomial_draw(int64_t n, double p,
                                       double (*u01)(uint64_t*),
                                       uint64_t *state);
#endif

static void pivot_hist_one_row(const double* p, mwSize d,
                               uint64_t K, uint64_t seed,
                               double* out /*size d*/)
{
    // Step 1: independent Poisson(K*p_i)
    int64_t S = 0;
    uint64_t st = seed;
    for (mwSize i=0;i<d;i++){
        const double mu = (double)K * p[i];
        const int64_t yi = (mu>0.0) ? binomcore_poisson_draw(mu, u01_open, &st) : 0;
        out[i] = (double)yi;
        S += yi;
    }

    if ((uint64_t)S == K) return;

    if ((uint64_t)S < K){
        // Need to add R = K - S counts drawn from categorical p
        const uint64_t R = K - (uint64_t)S;
        // Build a quick alias for p (one row)
        mxArray *prob_mx=NULL, *alias_mx=NULL;
        alias_build_arrays(p, d, &prob_mx, &alias_mx);
        const double  *q = mxGetPr(prob_mx);
        const int32_T *J = (const int32_T*)mxGetData(alias_mx);
        for (uint64_t t=0;t<R;t++){
            const mwSize k = alias_sample_one(q,J,d,&st);
            out[k] += 1.0;
        }
        mxDestroyArray(prob_mx);
        mxDestroyArray(alias_mx);
    }else{
        // Need to remove R = S - K counts proportionally to current out
        // Do exact multinomial thinning by sequential Hypergeometric:
        //   remove r_i ~ HyperGeom(remS, y_i, R_rem) in sequence
        uint64_t R = (uint64_t)S - K;
        int64_t remS = S;
        for (mwSize i=0;i<d && R>0;i++){
            const int64_t yi = (int64_t)(out[i] + 0.5);
            if (yi<=0) continue;
            // sample remove_i ~ HyperGeom(remS, yi, R)
            // via binomial equivalence in sequential scheme:
            // remove_i ~ Binomial(yi, (double)R / remS)
            const double pr = ((double)R) / (double)remS;
            int64_t rem_i = (pr<=0) ? 0 : ( (pr>=1)? yi
                                   : binomcore_binomial_draw(yi, pr, u01_open, &st) );
            if (rem_i > yi) rem_i = yi;
            out[i] = (double)(yi - rem_i);
            R     -= (uint64_t)rem_i;
            remS  -= yi;
        }
        // In the unlikely event of leftover 1–2 counts due to rounding,
        // remove them uniformly (keeps exact sum)
        while (R>0){
            // find next positive cell
            for (mwSize i=0;i<d && R>0;i++){
                if (out[i] > 0.0){ out[i] -= 1.0; --R; }
            }
        }
    }
}

static void pivot_hist_batch(const double* p, mwSize d,
                             uint64_t K, mwSize m, uint64_t seed,
                             double* H /*row-major*/)
{
    OMP_PRAGMA(omp parallel for schedule(static) if(m>1))
    for (ptrdiff_t r=0; r<(ptrdiff_t)m; ++r){
        uint64_t st = seed + (uint64_t)r * UINT64_C(1442695040888963407);
        double *row = H + (ptrdiff_t)r*(ptrdiff_t)d;
        for (mwSize j=0;j<d;j++) row[j] = 0.0;
        pivot_hist_one_row(p, d, K, st, row);
    }
}

// ================================================================
// mx helpers
// ================================================================
static mxArray* make_hist_matrix(mwSize m, mwSize d){
    // Return m-by-d double (MATLAB column-major). We fill row-major then copy.
    mxArray *A = mxCreateDoubleMatrix(m, d, mxREAL);
    // We'll copy from a row-major buffer in the run_*_mex wrappers.
    return A;
}
static void copy_rowmajor_to_mx(const double *rowMajor, mwSize m, mwSize d, mxArray *A){
    double *dst = mxGetPr(A);
    // Convert row-major m-by-d buffer to MATLAB column-major d blocks
    for (mwSize j=0;j<d;j++){
        const double *src_col = rowMajor + (ptrdiff_t)j;
        double *dst_col = dst + (ptrdiff_t)j*(ptrdiff_t)m;
        for (mwSize i=0;i<m;i++){
            dst_col[i] = src_col[(ptrdiff_t)i*(ptrdiff_t)d];
        }
    }
}

// ================================================================
// Public wrappers (return ready-to-hand-back mxArray*)
// ================================================================
static mxArray* run_cdf_smallk_mex(const double *p, mwSize d,
                                   uint64_t K, mwSize m, uint64_t seed)
{
    mxArray *A = make_hist_matrix(m,d);
    double *tmp = (double*)mxCalloc((size_t)m*(size_t)d, sizeof(double));
    cdf_smallk_hist(p,d,K,m,seed,tmp);
    copy_rowmajor_to_mx(tmp,m,d,A);
    mxFree(tmp);
    return A;
}

static mxArray* run_alias_hist_thin_mex(const double *q, const int32_T *J,
                                        mwSize d, uint64_t K, mwSize m,
                                        uint64_t seed)
{
    mxArray *A = make_hist_matrix(m,d);
    double *tmp = (double*)mxCalloc((size_t)m*(size_t)d, sizeof(double));
    alias_hist_thin(q,J,d,K,m,seed,tmp);
    copy_rowmajor_to_mx(tmp,m,d,A);
    mxFree(tmp);
    return A;
}

static mxArray* run_pivot_hist_batch_mex(const double *p, mwSize d,
                                         uint64_t K, mwSize m, uint64_t seed)
{
    mxArray *A = make_hist_matrix(m,d);
    double *tmp = (double*)mxCalloc((size_t)m*(size_t)d, sizeof(double));
    pivot_hist_batch(p,d,K,m,seed,tmp);
    copy_rowmajor_to_mx(tmp,m,d,A);
    mxFree(tmp);
    return A;
}

static mxArray* run_pivot_hist_one_mex(const double *p, mwSize d,
                                       uint64_t K, uint64_t seed)
{
    mxArray *A = make_hist_matrix(1,d);
    double *tmp = (double*)mxCalloc((size_t)d, sizeof(double));
    pivot_hist_one_row(p,d,K,seed,tmp);
    copy_rowmajor_to_mx(tmp,1,d,A);
    mxFree(tmp);
    return A;
}
