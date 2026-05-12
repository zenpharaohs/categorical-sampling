/* binomial_kernels_mex.c  (C-only; OpenMP; exact)
 *
 * Kernels:
 *   x = mex('btrd', n, p, m, seed)            // fast exact table inversion
 *   x = mex('btpe', n, p, m, seed)            // alias to btrd (kept for compatibility)
 *   x = mex('wait2', n, p, m, seed)           // waiting-times (exact), OpenMP
 *   x = mex('dev',  n, p, m, seed[, opts])    // Devroye Beta-pivot; early cutoffs
 *   st= mex('btrd_build', n, p);  x = mex('btrd_draw_b', st, m, seed);
 *
 * opts (optional struct for 'dev'):
 *   - kappa    : double, default 14.0   (tail bailout to wait2 when mu <= kappa*log1p(n)-1)
 *   - n_switch : double, default 2048   (when residual n <= n_switch -> finish via center-out inversion)
 *   - n_switch_auto: set n_switch=1 in opts to enable auto-scaling to n^(2/3) -- slower in practice
 *
 * Deterministic across thread counts (per-sample SplitMix64 substreams).
 *
 * Suggested build:
 *   mex -O 'CFLAGS=$CFLAGS -O3 -fopenmp -march=native -mtune=native' \
 *       'LDFLAGS=$LDFLAGS -fopenmp' binomial_kernels_mex.c
 */

 #include "mex.h"
 #include <math.h>
 #include <float.h>
 #include <stdint.h>
 #include <string.h>

 #ifdef _OPENMP
   #include <omp.h>
 #endif

 #ifndef OMP_CHUNK
 #define OMP_CHUNK 4096
 #endif

 /* ====================== RNG ====================== */
 static inline uint64_t splitmix64(uint64_t *s){
     uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
     z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
     z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
     return z ^ (z >> 31);
 }
 static inline double u01(uint64_t *s){                   /* (0,1), 53-bit */
     const uint64_t r = splitmix64(s);
     const uint64_t m = (r >> 11) | 1ULL;                 /* avoid exact 0 */
     return (double)m * (1.0/9007199254740992.0);         /* 2^53 */
 }
 static inline uint64_t u64(uint64_t *s){ return splitmix64(s); }

 /* ====================== Ziggurat Exp(1) ====================== */
 /* N=256 layers; R=7.697 tail threshold; ~99% fast-accept path.
  * Replaces -log(U) in both the standalone wait2 kernel and the dev
  * wait2 bailout.  Tables initialised once via a static flag.
  * Reference: Marsaglia & Tsang (2000), J. Stat. Soft. 5(8).
  */
 #define ZEXP_N 256
 static uint32_t zexp_ke[ZEXP_N];
 static double   zexp_we[ZEXP_N];
 static double   zexp_fe[ZEXP_N+1];
 static double   zexp_x [ZEXP_N+1];
 static int      zexp_inited = 0;

 static void zexp_init_(void){
     if (zexp_inited) return;
     const double R   = 7.69711747013104972;
     const double eR  = exp(-R);
     const double Ain = (1.0 - eR) / (double)ZEXP_N;
     zexp_x[0]  = R;
     zexp_fe[0] = eR;
     for (int i = 1; i <= ZEXP_N; ++i){
         double ei = eR + (double)i * Ain;
         if (ei > 1.0) ei = 1.0;
         zexp_x[i]  = -log(ei);
         zexp_fe[i] = ei;
     }
     for (int i = 0; i < ZEXP_N; ++i){
         double w = zexp_x[i] - zexp_x[i+1];
         zexp_we[i] = w * (1.0/4294967296.0);
         double r = zexp_fe[i+1] / zexp_fe[i];
         double t = r * 4294967296.0;
         zexp_ke[i] = (t >= 4294967295.0) ? 0xFFFFFFFFu : (uint32_t)t;
     }
     zexp_inited = 1;
 }

 /* Returns one Exp(1) deviate.  Fast path: no transcendentals. */
 static inline double expdev_zig_(uint64_t *s){
     zexp_init_();
     for (;;){
         uint64_t r = u64(s);
         uint32_t j = (uint32_t)(r & 0xFFFFFFFFu);
         int      i = (int)((r >> 32) & 0xFFu);
         if (j < zexp_ke[i])
             return zexp_x[i+1] + (double)j * zexp_we[i];   /* fast (~99%) */
         if (i == 0)
             return zexp_x[0] - log(u01(s));                 /* tail: rare  */
         double x = zexp_x[i+1] + (double)j * zexp_we[i];
         double y = zexp_fe[i+1] + (zexp_fe[i] - zexp_fe[i+1]) * u01(s);
         if (y <= exp(-x)) return x;                         /* exact test  */
     }
 }

 /* ====================== Helpers ====================== */
 static inline int is_string_scalar(const mxArray *a){
     return mxIsChar(a) && mxGetNumberOfElements(a) > 0;
 }
 static inline uint64_t read_seed_any(const mxArray *a){
     if (mxIsUint64(a)) {
         const uint64_t *sp = (const uint64_t*) mxGetData(a);
         return sp ? *sp : 0ULL;
     } else {
         double v = mxGetScalar(a);
         if (!(v==v)) v = 0.0;
         return (uint64_t) llround(v);
     }
 }
 static inline double logpmf_binom_at(size_t n, size_t k, double p){
     if (k > n) return -INFINITY;
     if (p <= 0.0) return (k==0)? 0.0 : -INFINITY;
     if (p >= 1.0) return (k==n)? 0.0 : -INFINITY;
     double q = 1.0 - p;
     double lg = lgamma((double)n + 1.0)
               - lgamma((double)k + 1.0)
               - lgamma((double)(n - k) + 1.0);
     return lg + (double)k*log(p) + (double)(n - k)*log(q);
 }

 /* ====================== fast table inversion (btrd/btpe) ====================== */
 static void build_cdf_scaled(size_t n, double pin, double **cdf_out, double *Z_out, int *flip_out){
     int flip = (pin > 0.5) ? 1 : 0;
     double p = flip ? (1.0 - pin) : pin;
     double q = 1.0 - p;

     if (p <= 0.0){
         double *cdf = (double*) mxCalloc(n+1, sizeof(double));
         cdf[0] = 1.0; for (size_t k=1;k<=n;++k) cdf[k] = 1.0;
         *cdf_out=cdf; *Z_out=1.0; *flip_out=flip; return;
     }
     if (p >= 1.0){
         double *cdf = (double*) mxCalloc(n+1, sizeof(double));
         for (size_t k=0;k<n;++k) cdf[k] = 0.0; cdf[n] = 1.0;
         *cdf_out=cdf; *Z_out=1.0; *flip_out=flip; return;
     }

     size_t m = (size_t) floor(((double)n + 1.0) * p);
     if (m > n) m = n;

     double *pmf = (double*) mxCalloc(n+1, sizeof(double));
     pmf[m] = 1.0;
     for (size_t k=m; k>=1; --k){
         pmf[k-1] = pmf[k] * ((double)k / (double)(n - k + 1)) * ( (1.0-p) / p );
         if (k==1) break;
     }
     for (size_t k=m; k<n; ++k){
         pmf[k+1] = pmf[k] * ((double)(n - k) / (double)(k + 1)) * ( p / (1.0-p) );
     }

     double *cdf = (double*) mxCalloc(n+1, sizeof(double));
     double s = 0.0;
     for (size_t k=0; k<=n; ++k){ s += pmf[k]; cdf[k] = s; }
     mxFree(pmf);

     *cdf_out = cdf; *Z_out = s; *flip_out = flip;
 }
 static inline size_t cdf_lower_bound(const double *cdf, size_t n, double x){
     size_t lo=0, hi=n+1;
     while (lo < hi){
         size_t mid = lo + ((hi - lo) >> 1);
         if (cdf[mid] >= x) hi = mid; else lo = mid + 1;
     }
     return (lo > n ? n : lo);
 }
 static void btrd_draw_fresh(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
     (void)nlhs;
     if (nrhs < 5) mexErrMsgIdAndTxt("binokernels:btrd_draw:nrhs","btrd: need n,p,m,seed");
     size_t n = (size_t) mxGetScalar(prhs[1]);
     double p  = mxGetScalar(prhs[2]);
     size_t m  = (size_t) mxGetScalar(prhs[3]);
     uint64_t seed = read_seed_any(prhs[4]);
     if (p < 0.0 || p > 1.0) mexErrMsgIdAndTxt("binokernels:btrd_draw:p","p in [0,1]");

     double *cdf; double Z; int flip;
     build_cdf_scaled(n, p, &cdf, &Z, &flip);

     plhs[0] = mxCreateNumericMatrix((mwSize)m, 1, mxUINT32_CLASS, mxREAL);
     uint32_t *out = (uint32_t*) mxGetData(plhs[0]);

 #ifdef _OPENMP
 #pragma omp parallel for schedule(dynamic, OMP_CHUNK)
 #endif
     for (ptrdiff_t i=0; i<(ptrdiff_t)m; ++i){
         uint64_t s = seed ^ (0xC6A4A7935BD1E995ULL * ((uint64_t)i + 1ULL));
         double t = u01(&s) * Z;
         size_t k = cdf_lower_bound(cdf, n, t);
         out[i] = flip ? (uint32_t)(n - k) : (uint32_t)k;
     }
     mxFree(cdf);
 }
 static void btrd_build_mex(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
     (void)nlhs;
     if (nrhs < 3) mexErrMsgIdAndTxt("binokernels:btrd_build:nrhs","btrd_build: need n,p");
     size_t n = (size_t) mxGetScalar(prhs[1]);
     double p  = mxGetScalar(prhs[2]);
     double *cdf; double Z; int flip;
     build_cdf_scaled(n, p, &cdf, &Z, &flip);
     const char *fn[]={"n","p","flip","Z","cdf"};
     plhs[0] = mxCreateStructMatrix(1,1,5,fn);
     mxSetField(plhs[0],0,"n",    mxCreateDoubleScalar((double)n));
     mxSetField(plhs[0],0,"p",    mxCreateDoubleScalar(p));
     mxSetField(plhs[0],0,"flip", mxCreateDoubleScalar((double)flip));
     mxSetField(plhs[0],0,"Z",    mxCreateDoubleScalar(Z));
     mxArray *cdf_mx = mxCreateDoubleMatrix((mwSize)(n+1), 1, mxREAL);
     memcpy(mxGetPr(cdf_mx), cdf, (n+1)*sizeof(double));
     mxSetField(plhs[0],0,"cdf", cdf_mx);
     mxFree(cdf);
 }
 static void btrd_draw_built(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
     (void)nlhs;
     if (nrhs < 4) mexErrMsgIdAndTxt("binokernels:btrd_draw_b:nrhs","btrd_draw_b: need state,m,seed");
     const mxArray *st = prhs[1];
     size_t m  = (size_t) mxGetScalar(prhs[2]);
     uint64_t seed = read_seed_any(prhs[3]);
     size_t n   = (size_t) mxGetScalar(mxGetField(st,0,"n"));
     int flip   = (int)      mxGetScalar(mxGetField(st,0,"flip"));
     double Z   =              mxGetScalar(mxGetField(st,0,"Z"));
     const mxArray *cdf_mx =   mxGetField(st,0,"cdf");
     if (!cdf_mx) mexErrMsgIdAndTxt("binokernels:btrd_draw_b:state","missing cdf");
     const double *cdf = mxGetPr(cdf_mx);

     plhs[0] = mxCreateNumericMatrix((mwSize)m, 1, mxUINT32_CLASS, mxREAL);
     uint32_t *out = (uint32_t*) mxGetData(plhs[0]);

 #ifdef _OPENMP
 #pragma omp parallel for schedule(dynamic, OMP_CHUNK)
 #endif
     for (ptrdiff_t i=0; i<(ptrdiff_t)m; ++i){
         uint64_t s = seed ^ (0x8F3F73B5CF1C9E3BULL * ((uint64_t)i + 1ULL));
         double t = u01(&s) * Z;
         size_t k = cdf_lower_bound(cdf, n, t);
         out[i] = flip ? (uint32_t)(n - k) : (uint32_t)k;
     }
 }

 /* ====================== center-out exact inversion (no build) ====================== */
 static uint32_t inv_center_out_one(size_t n, double p, uint64_t *s){
     if (p <= 0.0) return 0;
     if (p >= 1.0) return (uint32_t)n;
     int flip = 0; if (p > 0.5){ p = 1.0 - p; flip = 1; }
     double q = 1.0 - p;

     size_t m = (size_t) floor(((double)n + 1.0) * p);
     if (m > n) m = n;

     double pm = exp(logpmf_binom_at(n, m, p));

     double u  = u01(s);
     double sum = pm;
     if (u <= sum) return flip ? (uint32_t)(n - m) : (uint32_t)m;

     size_t kL = m, kR = m;
     double pL = pm, pR = pm;

     for (;;){
         if (kL > 0){
             pL *= ((double)kL / (double)(n - kL + 1)) * ((1.0 - p) / p);
             kL--;
             sum += pL;
             if (u <= sum) return flip ? (uint32_t)(n - kL) : (uint32_t)kL;
         }
         if (kR < n){
             pR *= ((double)(n - kR) / (double)(kR + 1)) * (p / (1.0 - p));
             kR++;
             sum += pR;
             if (u <= sum) return flip ? (uint32_t)(n - kR) : (uint32_t)kR;
         }
         if (kL==0 && kR==n) return flip ? 0U : (uint32_t)n; /* safety */
     }
 }

 /* ====================== wait2 (waiting-times) ====================== */
 /* Draw m IID Bin(n,p) via geometric inter-arrivals; per-thread streams */
 static void wait2_draw_mex(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
     (void)nlhs;
     if (nrhs < 5) mexErrMsgIdAndTxt("binokernels:wait2:nrhs","wait2: need n,p,m,seed");
     size_t n = (size_t) mxGetScalar(prhs[1]);
     double p  = mxGetScalar(prhs[2]);
     size_t m  = (size_t) mxGetScalar(prhs[3]);
     uint64_t seed = read_seed_any(prhs[4]);
     if (p < 0.0 || p > 1.0) mexErrMsgIdAndTxt("binokernels:wait2:p","p in [0,1]");

     int flip = 0; if (p > 0.5){ p = 1.0 - p; flip = 1; }
     if (p <= 0.0){
         plhs[0] = mxCreateNumericMatrix((mwSize)m,1,mxUINT32_CLASS,mxREAL);
         uint32_t *o = (uint32_t*)mxGetData(plhs[0]);
 #ifdef _OPENMP
 #pragma omp parallel for schedule(static)
 #endif
         for (ptrdiff_t i=0;i<(ptrdiff_t)m;++i) o[i] = flip ? (uint32_t)n : 0U;
         return;
     }
     if (p >= 1.0){
         plhs[0] = mxCreateNumericMatrix((mwSize)m,1,mxUINT32_CLASS,mxREAL);
         uint32_t *o = (uint32_t*)mxGetData(plhs[0]);
 #ifdef _OPENMP
 #pragma omp parallel for schedule(static)
 #endif
         for (ptrdiff_t i=0;i<(ptrdiff_t)m;++i) o[i] = flip ? 0U : (uint32_t)n;
         return;
     }

     const double invlam = 1.0 / (-log1p(-p));  /* E ~ Exp(1)/lambda with lambda=-log(1-p) */
     zexp_init_();

     plhs[0] = mxCreateNumericMatrix((mwSize)m,1,mxUINT32_CLASS,mxREAL);
     uint32_t *out = (uint32_t*)mxGetData(plhs[0]);

 #ifdef _OPENMP
 #pragma omp parallel for schedule(dynamic, OMP_CHUNK)
 #endif
     for (ptrdiff_t i=0; i<(ptrdiff_t)m; ++i){
         uint64_t s = seed ^ (0xA0761D6478BD642FULL * ((uint64_t)i + 1ULL));
         size_t t = 0; uint32_t k = 0;
         while (t < n){
             double e = expdev_zig_(&s);           /* Exp(1) via ziggurat */
             size_t G = (size_t) floor( e * invlam );
             t += G + 1;
             if (t <= n) k++;
         }
         out[i] = flip ? ((uint32_t)n - k) : k;
     }
 }

 /* ====================== Devroye Beta-pivot with early cutoff ====================== */

 /* --- Normal via ratio-of-uniforms (Leva 1992, ACM TOMS 18:449-453) ---
  *
  * Replaces Box-Muller polar + normcache_t.  Advantages:
  *   - No sqrt() ever required (Box-Muller needs one per accepted pair).
  *   - ~76% fast-accept path: 2 uniforms + a few multiplies, no log.
  *   - ~21% mid path:         2 uniforms + 1 log.
  *   - ~3%  fast-reject:      2 uniforms, no transcendentals.
  *   - Each call independent; no pairing/caching needed.
  *
  * Constants: quadratic squeeze tangent to the exact RoU boundary
  *   v^2 = -4u^2 ln(u)  at  (s,t) = (0.449871, 0.386595).
  * Scale 1.7156 = 2*sqrt(2/e) covers the full proposal interval for v.
  */
 static inline double normal_leva_(uint64_t *s) {
     for (;;) {
         double u = u01(s);
         double v = (u01(s) - 0.5) * 1.7156;   /* v ~ U(-sqrt(2/e), sqrt(2/e)) */
         double x = u - 0.449871;
         double y = fabs(v) + 0.386595;  /* y = |v| - t, t = -0.386595 in Leva */
         double q = x*x + y*(0.19600*y - 0.25472*x);
         if (q < 0.27597) return v / u;          /* fast accept (~76%) */
         if (q > 0.27846) continue;              /* fast reject  (~3%) */
         if (v*v < -4.0 * log(u) * u*u) return v / u;  /* exact (~21%) */
     }
 }

 /* --- Marsaglia-Tsang Gamma(a) using normal_leva_ ---
  *
  * For a >= 1 (always true in dev pivot loop): one Leva normal + one uniform
  * in the fast path (~90% of draws).  The a < 1 branch is kept for
  * generality but is never reached from dev.
  */
 static inline double gamma_mt(double a, uint64_t *s) {
     if (a <= 0.0) return 0.0;
     if (a < 1.0) {
         double g = gamma_mt(a + 1.0, s);
         double u = u01(s);
         return g * pow(u, 1.0 / a);
     }
     const double d = a - 1.0/3.0;
     const double c = 1.0 / sqrt(9.0*d);
     for (;;) {
         double z = normal_leva_(s);
         double x = 1.0 + c*z;
         if (x <= 0.0) continue;
         x = x*x*x;
         double u = u01(s);
         if (u < 1.0 - 0.0331*z*z*z*z) return d*x;
         if (log(u) < 0.5*z*z + d*(1.0 - x + log(x))) return d*x;
     }
 }

 /* dev options:
  *   kappa    : tail-bailout threshold multiplier (default 14.0)
  *   n_switch : residual-n threshold for center-out inversion.
  *              Default AUTO: max(2048, floor(n^(2/3))), which achieves
  *              the O(log log n) pivot-step bound of Devroye's algorithm.
  *              Set explicitly to override (mainly for benchmarking).
  */
 typedef struct { double kappa; size_t n_switch; int n_switch_auto; } dev_opts_t;
 static dev_opts_t parse_dev_opts(int nrhs, const mxArray *prhs[]){
     dev_opts_t o; o.kappa = 14.0; o.n_switch = 2048; o.n_switch_auto = 0;
     if (nrhs >= 6 && mxIsStruct(prhs[5])){
         mxArray *f;
         f = mxGetField(prhs[5],0,"kappa");
         if (f && mxIsDouble(f)) { double v = mxGetScalar(f); if (isfinite(v) && v>0) o.kappa = v; }
         f = mxGetField(prhs[5],0,"n_switch");
         if (f && mxIsDouble(f)) { double v = mxGetScalar(f); if (v>=1){ o.n_switch = (size_t)llround(v); o.n_switch_auto = 0; } }
     }
     return o;
 }

 static uint32_t devroye_hybrid_one(size_t n0, double p0, uint64_t *s, const dev_opts_t *opt){
     if (p0 <= 0.0) return 0;
     if (p0 >= 1.0) return (uint32_t)n0;
     int flip = 0; if (p0 > 0.5){ p0 = 1.0 - p0; flip = 1; }

     size_t n = n0;
     double  p = p0;
     uint32_t z = 0U;

     const double kappa = opt->kappa;
     /* n_switch: auto-scale to n^(2/3) to achieve O(log log n) pivot steps.
      * Floor at 2048 to keep center-out inversion cost bounded.
      * User override (n_switch_auto==0) skips this. */
     const size_t n_switch = opt->n_switch_auto
         ? (size_t) fmax(2048.0, floor(pow((double)n0, 2.0/3.0)))
         : opt->n_switch;

     for (;;){
         if (n == 0 || p <= 0.0){ break; }

         /* ---- early finish via center-out inversion when residual n small ---- */
         if (n <= n_switch){
             uint32_t rest = inv_center_out_one(n, p, s);
             z += rest;
             break;
         }

         /* ---- tail bailout (exact) when mu small ---- */
         double mu = n * (p <= 0.5 ? p : (1.0-p));
         double thresh = kappa*log1p((double)n) - 1.0;
         if (thresh < 0.0) thresh = 0.0;
         if (mu <= thresh){
             /* finish via waiting-times — symmetrize p first.
              * Pivot updates can push p back above 0.5; using raw p > 0.5
              * gives invlam ~ 0, G ~ 0 every step, and O(n) iterations.
              * Flip to p_w = min(p,1-p) and adjust the count. */
             int fw = (p > 0.5) ? 1 : 0;
             double pw = fw ? (1.0 - p) : p;
             const double invlam = 1.0 / (-log1p(-pw));
             size_t t = 0, k = 0;
             while (t < n){
                 double e = expdev_zig_(s);        /* Exp(1) via ziggurat */
                 size_t G = (size_t) floor( e * invlam );
                 t += G + 1;
                 if (t <= n) k++;
             }
             z += fw ? (uint32_t)(n - k) : (uint32_t)k;
             break;
         }

         /* ---- Devroye pivot step ---- */
         size_t q = (size_t) ceil( (double)n * p );
         if (q < 1) q = 1; if (q > n) q = n;

         double x;
         if (q == 1){
             x = 1.0 - pow(u01(s), 1.0/ (double)n);           /* Beta(1,n) */
         } else if (q == n){
             x = pow(u01(s), 1.0/ (double)n);                 /* Beta(n,1) */
         } else {
             double u = gamma_mt((double)q, s);
             double v = gamma_mt((double)(n + 1 - q), s);
             x = u / (u + v);
             if (!(x>0.0 && x<1.0)){ x = fmin(fmax(x, DBL_MIN), 1.0-DBL_MIN); }
         }

         if (x <= p){
             z += (uint32_t) q;
             n  -= q;
             double den = 1.0 - x; if (den <= 0.0) den = DBL_MIN;
             p  = (p - x) / den;
             if (p < 0.0) p = 0.0; if (p > 1.0) p = 1.0;
         } else {
             n  = q - 1;
             double den = x; if (den <= 0.0) den = DBL_MIN;
             p  = p / den;
             if (p < 0.0) p = 0.0; if (p > 1.0) p = 1.0;
         }
     }
     return flip ? (uint32_t)(n0 - z) : z;
 }

 static void dev_draw(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
     (void)nlhs;
     if (nrhs < 5) mexErrMsgIdAndTxt("binokernels:dev:nrhs","dev: need n,p,m,seed[, opts]");
     size_t n = (size_t) mxGetScalar(prhs[1]);
     double p  = mxGetScalar(prhs[2]);
     size_t m  = (size_t) mxGetScalar(prhs[3]);
     uint64_t seed = read_seed_any(prhs[4]);
     if (p < 0.0 || p > 1.0) mexErrMsgIdAndTxt("binokernels:dev:p","p in [0,1]");

     dev_opts_t opt = parse_dev_opts(nrhs, prhs);

     plhs[0] = mxCreateNumericMatrix((mwSize)m, 1, mxUINT32_CLASS, mxREAL);
     uint32_t *out = (uint32_t*) mxGetData(plhs[0]);

 #ifdef _OPENMP
 #pragma omp parallel for schedule(dynamic, OMP_CHUNK)
 #endif
     for (ptrdiff_t i=0; i<(ptrdiff_t)m; ++i){
         uint64_t s = seed ^ (0xD1B54A32D192ED03ULL * ((uint64_t)i + 1ULL));
         out[i] = devroye_hybrid_one(n, p, &s, &opt);
     }
 }

 /* ====================== dispatch ====================== */
 void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
 {
     if (nrhs < 1 || !is_string_scalar(prhs[0])){
         mexErrMsgIdAndTxt("binokernels:usage",
             "Usage: binomial_kernels_mex(cmd, ...)\n"
             "  btrd | btpe | wait2 | dev | btrd_build | btrd_draw_b\n"
             "  dev accepts optional opts struct: kappa, n_switch");
     }
     char cmd[64]; mxGetString(prhs[0], cmd, sizeof(cmd));

     if (!strcmp(cmd,"btrd")  || !strcmp(cmd,"btrd_draw"))  { btrd_draw_fresh(nlhs, plhs, nrhs, prhs); return; }
     if (!strcmp(cmd,"btpe")  || !strcmp(cmd,"btpe_draw"))  { btrd_draw_fresh(nlhs, plhs, nrhs, prhs); return; }
     if (!strcmp(cmd,"btrd_build")) { btrd_build_mex(nlhs, plhs, nrhs, prhs); return; }
     if (!strcmp(cmd,"btrd_draw_b")){ btrd_draw_built(nlhs, plhs, nrhs, prhs); return; }
     if (!strcmp(cmd,"wait2") || !strcmp(cmd,"wait2_draw")) { wait2_draw_mex(nlhs, plhs, nrhs, prhs); return; }
     if (!strcmp(cmd,"dev")   || !strcmp(cmd,"dev_draw"))   { dev_draw(nlhs, plhs, nrhs, prhs); return; }

     mexErrMsgIdAndTxt("binokernels:cmd","Unknown cmd '%s'", cmd);
 }

