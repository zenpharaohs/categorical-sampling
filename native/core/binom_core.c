#include "binom_core.h"
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stddef.h>

/* ============================================================
 * Helpers
 * ============================================================ */
static inline double clamp01_open(double u){
    if (!(u>0.0)) u = DBL_MIN;
    if (u>=1.0)   u = 1.0 - DBL_MIN;
    return u;
}

/* ============================================================
 * Exact center–out inversion (always exact)
 * ============================================================ */
size_t binom_centerout_core(size_t n, double p,
                            double (*U01)(void*), void* ctx)
{
    if (p <= 0.0) return 0;
    if (p >= 1.0) return n;

    int flip = 0;
    if (p > 0.5){ p = 1.0 - p; flip = 1; }
    const double q = 1.0 - p;

    size_t m = (size_t) floor(((double)n + 1.0) * p);
    if (m > n) m = n;

    /* log pmf at mode */
    double lg = lgamma((double)n + 1.0)
              - lgamma((double)m + 1.0)
              - lgamma((double)(n - m) + 1.0);
    double pm = exp(lg + (double)m*log(p) + (double)(n - m)*log(q));

    double u  = clamp01_open(U01(ctx));
    double sum = pm;
    if (u <= sum) return flip ? (n - m) : m;

    size_t kL = m, kR = m;
    double pL = pm, pR = pm;
    for(;;){
        if (kL > 0){
            pL *= ((double)kL / (double)(n - kL + 1)) * (q / p);
            kL--;
            sum += pL;
            if (u <= sum) return flip ? (n - kL) : kL;
        }
        if (kR < n){
            pR *= ((double)(n - kR) / (double)(kR + 1)) * (p / q);
            kR++;
            sum += pR;
            if (u <= sum) return flip ? (n - kR) : kR;
        }
    }
}

/* ============================================================
 * Ziggurat for Exp(1)   (Marsaglia–Tsang style, N=256)
 *  - Fast accept branch uses only integer ops and mul/add.
 *  - Slow branch uses exp/log rarely.
 *  - Deterministic via caller-supplied U64/U01 callbacks.
 * ============================================================ */
#define ZEXP_N 256

static uint32_t zexp_ke[ZEXP_N];   /* accept thresholds */
static double   zexp_we[ZEXP_N];   /* widths scaled by 2^-32 */
static double   zexp_fe[ZEXP_N+1]; /* e^{-x[i]} at breakpoints */
static double   zexp_x [ZEXP_N+1]; /* breakpoints x[0]=R > ... > x[N]=0 */
static int      zexp_inited = 0;

static void zexp_init_(void){
    if (zexp_inited) return;

    /* Choose R so tail work is well balanced (canonical value). */
    const double R = 7.69711747013104972;

    /* Build x so that each layer i has equal area between x[i]..x[i-1].
       For Exp(1): area(x[i], x[i-1]) = e^{-x[i]} - e^{-x[i-1]}.
       Set e^{-x[i]} linearly spaced from e^{-R} to 1. */
    const double eR  = exp(-R);
    const double Ain = (1.0 - eR) / (double)ZEXP_N;

    zexp_x[0] = R;
    zexp_fe[0] = eR;
    for (int i=1; i<=ZEXP_N; ++i){
        double ei = eR + (double)i * Ain;           /* e^{-x[i]} */
        if (ei > 1.0) ei = 1.0;
        zexp_x[i]  = -log(ei);
        zexp_fe[i] = ei;
    }
    /* Now x[0]=R > x[1] > ... > x[N]=0 and fe[i]=exp(-x[i]) increasing */

    for (int i=0; i<ZEXP_N; ++i){
        /* width of layer i is w = x[i] - x[i+1] (>0) */
        const double w = zexp_x[i] - zexp_x[i+1];
        zexp_we[i] = w * (1.0/4294967296.0);  /* scale by 2^-32 */

        /* Accept-threshold: fraction of the base rectangle under the curve.
           Height ratio fe[i+1]/fe[i] works for Exp(1). */
        double r = zexp_fe[i+1] / zexp_fe[i];
        if (r <= 0.0) zexp_ke[i] = 0;
        else {
            double t = r * 4294967296.0;     /* 2^32 */
            if (t >= 4294967295.0) zexp_ke[i] = 0xFFFFFFFFu;
            else                   zexp_ke[i] = (uint32_t)t;
        }
    }
    zexp_inited = 1;
}

static inline double expdev_ziggurat(binom_U01_f U01, binom_U64_f U64, void* ctx){
    zexp_init_();
    for (;;){
        uint64_t r = U64(ctx);
        uint32_t j = (uint32_t)(r & 0xFFFFFFFFu);
        int      i = (int)((r >> 32) & 0xFFu);   /* 0..255 */

        /* Fast accept region under the left rectangle slab. */
        if (j < zexp_ke[i]){
            /* Return a point uniformly over the base of the slab. */
            return zexp_x[i+1] + (double)j * zexp_we[i];
        }

        /* Slow branch: sample inside wedge beneath e^{-x} over [x[i+1], x[i]]. */
        double x  = zexp_x[i+1] + (double)j * zexp_we[i];
        if (i == 0){
            /* Top layer: memoryless tail */
            return zexp_x[0] - log( clamp01_open(U01(ctx)) );
        } else {
            /* Draw y uniformly in [fe[i+1], fe[i]] and accept if y <= e^{-x}. */
            double y = zexp_fe[i+1] + (zexp_fe[i] - zexp_fe[i+1]) * clamp01_open(U01(ctx));
            if (y <= exp(-x)) return x;
            /* else retry */
        }
    }
}

/* ============================================================
 * Waiting-times exact Binomial via Exp(1) inter-arrivals
 * ============================================================ */
size_t binom_wait2_core(size_t n, double p,
                        double   (*U01)(void*),
                        uint64_t (*U64)(void*),
                        void* ctx)
{
    if (p <= 0.0) return 0;
    if (p >= 1.0) return n;

    int flip = 0;
    if (p > 0.5){ p = 1.0 - p; flip = 1; }

    /* Poisson rate for waiting-times construction */
    const double lam = -log1p(-p);
    const double invlam = 1.0 / lam;

    size_t t = 0, k = 0;
    while (t < n){
        /* One Exp(1) draw via Ziggurat (fast), with rare fallback exp/log inside. */
        double E = expdev_ziggurat(U01, U64, ctx);
        size_t G = (size_t)(E * invlam);
        t += G + 1;
        if (t <= n) ++k;
    }
    return flip ? (n - k) : k;
}
