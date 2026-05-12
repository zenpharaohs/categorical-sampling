// multinomial_kernels_mex.c
// Unified kernels MEX: alias build/sample + exact histogram via binomial cascade.
// No external Poisson helpers needed. Pulls binomial core inline via include.
//
// Commands:
//   st = multinomial_kernels_mex('build', p)
//   cats = multinomial_kernels_mex('sample_b', st, N, seed)
//   H    = multinomial_kernels_mex('hist_b',   st, K, seed)   // 1-by-d histogram for one row
//
// Notes:
//   - 'hist_b' uses an exact binomial cascade: for j=1..d-1,
//       x_j ~ Binomial(K_rem, p_j / p_rem), then K_rem -= x_j.
//     This is exact for Multinomial(K, p), and fast using center-out inversion.
//   - We #include "binom_core.c" so we can call binom_centerout_core() directly.

#include "mex.h"
#include "matrix.h"
#include <stdint.h>
#include <math.h>
#include <string.h>

// ------------------------------------------------------------
// RNG: SplitMix64 (tiny, high quality) + U(0,1) 53-bit
// ------------------------------------------------------------
static inline uint64_t splitmix64_next(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline double u01_53(uint64_t *s) {
    // take top 53 bits -> [0,1)
    return ( (splitmix64_next(s) >> 11) ) * (1.0 / 9007199254740992.0); // 2^53
}

// ------------------------------------------------------------
// Pull in the exact binomial cores (center-out inversion, etc.)
//   - binom_centerout_core(n,p,U01,ctx) is all we need.
// ------------------------------------------------------------
#include "binom_core.c"

// Adapter so we can pass SplitMix64 as U01(void*)
static double U01_from_splitmix_ctx(void *ctx) {
    uint64_t *state = (uint64_t*)ctx;
    return u01_53(state);
}

// ------------------------------------------------------------
// Alias table build (Vose) + pack/unpack state struct
// ------------------------------------------------------------
typedef struct {
    mwSize    d;
    double   *p;    // original p (normalized)
    double   *q;    // alias probs (size d)
    int32_T  *J;    // alias indices (size d, 0-based)
} alias_state_view;

static void scan_prefix(const double *p, mwSize d, double *cdf) {
    double s = 0.0;
    for (mwSize i=0;i<d;i++){ s += p[i]; cdf[i] = s; }
}

static void alias_build_arrays(const double *p_in, mwSize d,
                               mxArray **p_out, mxArray **q_out, mxArray **J_out)
{
    // normalize p to sum 1, emit normalized p as well
    double sum = 0.0;
    for (mwSize i=0;i<d;i++) sum += p_in[i];
    if (!(sum > 0.0)) mexErrMsgIdAndTxt("multinom:alias","p must have positive sum.");

    mxArray *p_mx = mxCreateDoubleMatrix(1, d, mxREAL);
    mxArray *q_mx = mxCreateDoubleMatrix(1, d, mxREAL);
    mxArray *J_mx = mxCreateNumericMatrix(1, d, mxINT32_CLASS, mxREAL);

    double  *p = mxGetPr(p_mx);
    double  *q = mxGetPr(q_mx);
    int32_T *J = (int32_T*)mxGetData(J_mx);

    // scale a_i = p_i * d
    double *a = (double*)mxCalloc(d, sizeof(double));
    mwSize *small = (mwSize*)mxCalloc(d, sizeof(mwSize));
    mwSize *large = (mwSize*)mxCalloc(d, sizeof(mwSize));
    mwSize nsmall=0, nlarge=0;

    for (mwSize i=0;i<d;i++){
        p[i] = p_in[i] / sum;
        double ai = p[i] * (double)d;
        a[i] = ai;
        if (ai < 1.0) small[nsmall++] = i;
        else          large[nlarge++] = i;
    }

    while (nsmall && nlarge) {
        mwSize i = small[--nsmall];
        mwSize j = large[--nlarge];
        q[i] = a[i];
        J[i] = (int32_T)j;
        a[j] = (a[j] + a[i]) - 1.0;
        if (a[j] < 1.0) small[nsmall++] = j;
        else            large[nlarge++] = j;
    }
    while (nlarge) {
        mwSize j = large[--nlarge];
        q[j] = 1.0; J[j] = (int32_T)j;
    }
    while (nsmall) {
        mwSize i = small[--nsmall];
        q[i] = 1.0; J[i] = (int32_T)i;
    }

    mxFree(a); mxFree(small); mxFree(large);

    *p_out = p_mx; *q_out = q_mx; *J_out = J_mx;
}

static mxArray* pack_state_struct(mxArray *p_mx, mxArray *q_mx, mxArray *J_mx)
{
    const char* fns[] = {"p","prob","alias","d"};
    mxArray *st = mxCreateStructMatrix(1,1,4,fns);
    mxSetFieldByNumber(st,0,0,p_mx);
    mxSetFieldByNumber(st,0,1,q_mx);
    mxSetFieldByNumber(st,0,2,J_mx);
    mxArray *dval = mxCreateDoubleScalar( (double) mxGetNumberOfElements(p_mx) );
    mxSetFieldByNumber(st,0,3,dval);
    return st;
}

static void unpack_state_struct(const mxArray *st, alias_state_view *v)
{
    const mxArray *p_mx = mxGetField(st,0,"p");
    const mxArray *q_mx = mxGetField(st,0,"prob");
    const mxArray *J_mx = mxGetField(st,0,"alias");
    if (!p_mx || !q_mx || !J_mx)
        mexErrMsgIdAndTxt("multinom:state","state missing fields p/prob/alias.");

    v->d = (mwSize) mxGetNumberOfElements(p_mx);
    v->p = mxGetPr(p_mx);
    v->q = mxGetPr(q_mx);
    v->J = (int32_T*) mxGetData(J_mx);
}

// ------------------------------------------------------------
// sample_b: categories via alias (1-by-N int32 in 1..d)
// ------------------------------------------------------------
static mxArray* run_sample_b(const alias_state_view *v, mwSize N, uint64_t seed)
{
    mxArray *out = mxCreateNumericMatrix(1, N, mxINT32_CLASS, mxREAL);
    int32_T *cats = (int32_T*)mxGetData(out);
    double  *q = v->q;
    int32_T *J = v->J;
    mwSize   d = v->d;
    uint64_t s = seed;

    for (mwSize i=0;i<N;i++){
        // pick column
        double u  = u01_53(&s);
        mwSize k  = (mwSize) ( (double)d * u );
        if (k >= d) k = d-1;
        // accept or alias
        if (u01_53(&s) < q[k]) cats[i] = (int32_T)k + 1;
        else                   cats[i] = J[k] + 1;
    }
    return out;
}

// ------------------------------------------------------------
// hist_b: exact multinomial histogram via binomial cascade
//   H(1..d-1) ~ Binomial cascade; H_d = remainder.
// ------------------------------------------------------------
static mxArray* run_hist_b(const alias_state_view *v, uint64_t K, uint64_t seed)
{
    mwSize d = v->d;
    mxArray *Hmx = mxCreateDoubleMatrix(1, d, mxREAL);
    double *H = mxGetPr(Hmx);
    if (K == 0) return Hmx;

    uint64_t rem = K;
    double prem = 0.0;
    for (mwSize j=0;j<d;j++) prem += v->p[j];

    if (!(prem > 0.0)) mexErrMsgIdAndTxt("multinom:hist_b","p sum must be positive.");

    uint64_t s = seed;
    double sum_left = prem;

    for (mwSize j=0; j+1 < d; j++) {
        double pj = v->p[j];
        if (rem == 0 || sum_left <= 0.0 || pj <= 0.0) { H[j] = 0.0; sum_left -= pj; continue; }
        double pcond = pj / sum_left;
        if (pcond <= 0.0) { H[j] = 0.0; sum_left -= pj; continue; }
        if (pcond >= 1.0) {
            // everything left goes here (rare but handle)
            H[j] = (double)rem;
            for (mwSize t=j+1; t<d; t++) H[t] = 0.0;
            return Hmx;
        }
        uint64_t x = binom_centerout_core(rem, pcond, &U01_from_splitmix_ctx, &s);
        H[j] = (double)x;
        rem -= x;
        sum_left -= pj;
    }
    H[d-1] = (double)rem;
    return Hmx;
}

// ------------------------------------------------------------
// mexFunction: dispatcher
// ------------------------------------------------------------
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs < 1 || !mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt("multinom:usage",
            "Usage:\n  st = multinomial_kernels_mex('build', p)\n"
            "  cats = multinomial_kernels_mex('sample_b', st, N, seed)\n"
            "  H = multinomial_kernels_mex('hist_b', st, K, seed)\n");
    }

    char cmd[64];
    mxGetString(prhs[0], cmd, sizeof(cmd));

    // ---------------- build ----------------
    if (strcmp(cmd,"build")==0) {
        if (nrhs != 2) mexErrMsgIdAndTxt("multinom:build","build needs p.");
        const mxArray *p_in = prhs[1];
        if (!mxIsDouble(p_in) || mxIsComplex(p_in) ||
            (mxGetM(p_in) != 1 && mxGetN(p_in) != 1))
            mexErrMsgIdAndTxt("multinom:build","p must be a real vector.");

        mwSize d = (mwSize)mxGetNumberOfElements(p_in);
        if (d == 0) mexErrMsgIdAndTxt("multinom:build","p must be nonempty.");

        mxArray *p_mx = NULL, *q_mx = NULL, *J_mx = NULL;
        alias_build_arrays(mxGetPr(p_in), d, &p_mx, &q_mx, &J_mx);
        plhs[0] = pack_state_struct(p_mx, q_mx, J_mx);
        return;
    }

    // everything else needs a state
    if (nrhs < 2 || !mxIsStruct(prhs[1])) {
        mexErrMsgIdAndTxt("multinom:state","Second arg must be a state from 'build'.");
    }
    alias_state_view st;
    unpack_state_struct(prhs[1], &st);

    // ---------------- sample_b ----------------
    if (strcmp(cmd,"sample_b")==0) {
        if (nrhs != 4) mexErrMsgIdAndTxt("multinom:sample_b","sample_b needs st, N, seed.");
        if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || mxGetNumberOfElements(prhs[2])!=1)
            mexErrMsgIdAndTxt("multinom:sample_b","N must be scalar.");
        if (!mxIsUint64(prhs[3]) || mxGetNumberOfElements(prhs[3])!=1)
            mexErrMsgIdAndTxt("multinom:sample_b","seed must be uint64.");

        double Nd = mxGetScalar(prhs[2]);
        if (!(Nd >= 0.0)) mexErrMsgIdAndTxt("multinom:sample_b","N must be >= 0.");
        mwSize N = (mwSize)Nd;
        uint64_t seed = *( (const uint64_T*) mxGetData(prhs[3]) );
        plhs[0] = run_sample_b(&st, N, seed);
        return;
    }

    // ---------------- hist_b ----------------
    if (strcmp(cmd,"hist_b")==0) {
        if (nrhs != 4) mexErrMsgIdAndTxt("multinom:hist_b","hist_b needs st, K, seed.");
        if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || mxGetNumberOfElements(prhs[2])!=1)
            mexErrMsgIdAndTxt("multinom:hist_b","K must be scalar.");
        if (!mxIsUint64(prhs[3]) || mxGetNumberOfElements(prhs[3])!=1)
            mexErrMsgIdAndTxt("multinom:hist_b","seed must be uint64.");

        double Kd = mxGetScalar(prhs[2]);
        if (!(Kd >= 0.0)) mexErrMsgIdAndTxt("multinom:hist_b","K must be >= 0.");
        uint64_t K = (uint64_t) floor(Kd + 0.5);
        uint64_t seed = *( (const uint64_T*) mxGetData(prhs[3]) );
        plhs[0] = run_hist_b(&st, K, seed);
        return;
    }

    mexErrMsgIdAndTxt("multinom:cmd","Unknown command: %s", cmd);
}
