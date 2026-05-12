
// randmix_mex.c
// Categorical sampler using an alias table with optional replication.
// Usage:
//   idx = randmix_mex(p, m)
//   idx = randmix_mex(p, m, seed)
//   idx = randmix_mex(p, m, seed, R)
//     p: Kx1 or 1xK double (sum to 1; will be normalized)
//     m: number of samples (scalar)
//     seed: uint64 scalar (optional; default fixed value)
//     R: replication factor (optional int32/double scalar; default auto)
//        Auto default targets ~8MB L3 cache: R = floor(8MB / (K*12))
//        clamped to [1, 65536].  Higher R -> fewer alias lookups needed
//        (unsaturated fraction ~1/(e*R)) -> approaches cost of one uniform draw.
// Returns: int32 vector of length m with 1-based indices.
//
// Build example (MATLAB):
//   mex -O CFLAGS="$CFLAGS -O3 -fopenmp -march=native -mtune=native" \
//       LDFLAGS="$LDFLAGS -fopenmp" randmix_mex.c
//
#include "sampler_core.h"

typedef struct {
    int     K;      // number of categories
    int     M;      // KR = 2^M (KR rounded up to next power of 2)
    double  R_eff;  // effective R = 2^M / K (double, not necessarily integer)
    double  inv_K;  // precomputed 1.0/K -- avoids integer divide in draw
    double *prob;   // prob table (0..1), length K
    int    *alias;  // alias indices (0..K-1)
} alias_table_t;

static alias_table_t build_alias(const double *p, int K, int R){
    alias_table_t T;
    T.K     = K;
    T.inv_K = 1.0 / (double)K;
    // Round KR up to next power of 2 so the draw uses a bit shift, not a float multiply.
    uint64_t KR_target = (uint64_t)K * (uint64_t)R;
    int M = 0;
    uint64_t KR = 1;
    while (KR < KR_target) { KR <<= 1; M++; }
    T.M     = M;
    T.R_eff = (double)KR / (double)K;   // effective R (>= requested R)
    T.prob  = (double*)mxCalloc(K, sizeof(double));
    T.alias = (int*)   mxCalloc(K, sizeof(int));

    double *scaled = (double*)mxCalloc(K, sizeof(double));
    int *small = (int*)mxCalloc(K, sizeof(int));
    int *large = (int*)mxCalloc(K, sizeof(int));
    int nsmall = 0, nlarge = 0;

    // Normalize and scale by K
    double sum = 0.0;
    for (int i=0;i<K;i++) sum += p[i];
    if (sum <= 0) {
        mexErrMsgIdAndTxt("randmix:badp","Sum(p) must be positive.");
    }
    for (int i=0;i<K;i++){
        scaled[i] = p[i]/sum * K;
        if (scaled[i] < 1.0) small[nsmall++] = i;
        else                 large[nlarge++] = i;
    }

    while (nsmall && nlarge){
        int s = small[--nsmall];
        int l = large[--nlarge];
        T.prob[s] = scaled[s];
        T.alias[s] = l;
        scaled[l] = (scaled[l] + scaled[s]) - 1.0;
        if (scaled[l] < 1.0) small[nsmall++] = l;
        else                 large[nlarge++] = l;
    }
    while (nlarge){
        int l = large[--nlarge];
        T.prob[l] = 1.0;
        T.alias[l] = l;
    }
    while (nsmall){
        int s = small[--nsmall];
        T.prob[s] = 1.0;
        T.alias[s] = s;
    }

    mxFree(scaled); mxFree(small); mxFree(large);
    return T;
}

// Draw using virtual replication with power-of-2 KR = 2^M.
// Hot path: one raw uint64, one bit shift, one float multiply (inv_K),
// one integer multiply-subtract.  No division.  No second uniform
// except for the ~1/(e*R_eff) fraction of partially-saturated slots.
static inline int alias_draw_rep(rng64_t *r, const alias_table_t *T){
    // Map raw integer to [0, 2^M) with a single bit shift -- exact, no bias.
    uint64_t raw = rng64_next_u64(r);
    int j   = (int)(raw >> (64 - T->M));
    // Decompose j into (rep, k): rep = floor(j/K), k = j mod K.
    // Avoid integer divide via precomputed inv_K.
    int rep = (int)((double)j * T->inv_K);
    int k   = j - rep * T->K;
    if (k >= T->K) { k -= T->K; rep++; }  // fix rare float overshoot
    double thresh = T->prob[k] * T->R_eff - rep;
    if (thresh >= 1.0) return k;           // slot fully saturated (common)
    if (thresh <= 0.0) return T->alias[k]; // slot fully aliased
    double u2 = rng64_uniform01(r);        // rare: ~1/(e*R_eff)
    return (u2 < thresh) ? k : T->alias[k];
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    if (nrhs < 2) mexErrMsgIdAndTxt("randmix:args","Need p, m [, seed]");
    const mxArray *p_arr = prhs[0];
    const mxArray *m_arr = prhs[1];
    const mxArray *seed_arr = (nrhs >= 3 ? prhs[2] : NULL);

    if (!mxIsDouble(p_arr)) mexErrMsgIdAndTxt("randmix:p","p must be double");
    double *p = mxGetPr(p_arr);
    int K = (int)(mxGetM(p_arr) * mxGetN(p_arr));
    if (K < 1) mexErrMsgIdAndTxt("randmix:p","p must be non-empty");

    if (!mxIsDouble(m_arr) || mxGetNumberOfElements(m_arr)!=1)
        mexErrMsgIdAndTxt("randmix:m","m must be a scalar double");
    int64_t m = (int64_t)mxGetScalar(m_arr);
    if (m < 0) mexErrMsgIdAndTxt("randmix:m","m must be nonnegative");

    uint64_t seed = 0x12345678abcdef00ULL;
    if (seed_arr){
        if (!mxIsUint64(seed_arr) || mxGetNumberOfElements(seed_arr)!=1)
            mexErrMsgIdAndTxt("randmix:seed","seed must be uint64 scalar");
        seed = *(uint64_t*)mxGetData(seed_arr);
    }

    // Replication factor R: with virtual replication the alias table stays
    // K entries regardless of R -- no memory scales with R.  R appears only
    // in draw arithmetic (one integer divide + multiply per draw).  The only
    // effect of R is reducing the fraction of draws needing a second uniform:
    // ~1/(e*R).  At R=1024 that is 0.04% -- negligible.  The default of 1024
    // is near-optimal on any machine; calibrate() can refine per K-band.
    int R;
    const mxArray *R_arr = (nrhs >= 4 ? prhs[3] : NULL);
    if (R_arr) {
        if (!mxIsNumeric(R_arr) || mxGetNumberOfElements(R_arr) != 1)
            mexErrMsgIdAndTxt("randmix:R", "R must be a numeric scalar");
        R = (int)mxGetScalar(R_arr);
        if (R < 1) R = 1;
    } else {
        R = 1024;   // default: ~0.04% of draws need second uniform
    }
    // Build alias table on K elements only (O(K), not O(K*R)).
    // KR is rounded up to next power of 2 inside build_alias for bit-shift draw.
    alias_table_t T = build_alias(p, K, R);

    // Output vector
    plhs[0] = mxCreateNumericMatrix(m, 1, mxINT32_CLASS, mxREAL);
    int32_t *out = (int32_t*)mxGetData(plhs[0]);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint64_t base = seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(tid+1));
        rng64_t rr; rng64_seed(&rr, base);

        #pragma omp for schedule(static)
        for (int64_t j=0;j<m;j++){
            out[j] = (int32_t)(alias_draw_rep(&rr, &T) + 1);  // MATLAB 1-based
        }
    }

    mxFree(T.prob);
    mxFree(T.alias);
}
