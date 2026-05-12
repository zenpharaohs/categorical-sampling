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

static double   zexp_x [ZEXP_N+1]; /* breakpoints x[0]=0 < ... < x[N]=R */
static double   zexp_w [ZEXP_N];   /* interval widths */
static double   zexp_f [ZEXP_N+1]; /* e^{-x[i]} */
static double   zexp_r [ZEXP_N];   /* fast accept ratios f[i+1]/f[i] */
static double   zexp_tail = 0.0;   /* P(Exp(1) >= R) */
static int      zexp_inited = 0;

static void zexp_init_(void){
    if (zexp_inited) return;

    const double R = 7.69711747013104972;
    double lo = 0.0, hi = 0.01;
    for (;;) {
        double x = 0.0;
        for (int i=0; i<ZEXP_N; ++i) x += hi * exp(x);
        if (x >= R) break;
        hi *= 2.0;
    }
    for (int it=0; it<80; ++it) {
        double a = 0.5 * (lo + hi);
        double x = 0.0;
        for (int i=0; i<ZEXP_N; ++i) x += a * exp(x);
        if (x < R) lo = a; else hi = a;
    }
    const double A = 0.5 * (lo + hi);

    zexp_x[0] = 0.0;
    zexp_f[0] = 1.0;
    for (int i=0; i<ZEXP_N; ++i) {
        zexp_x[i+1] = zexp_x[i] + A * exp(zexp_x[i]);
        zexp_w[i] = zexp_x[i+1] - zexp_x[i];
        zexp_f[i+1] = exp(-zexp_x[i+1]);
        zexp_r[i] = zexp_f[i+1] / zexp_f[i];
    }
    zexp_tail = exp(-R);
    zexp_inited = 1;
}

static inline double expdev_ziggurat(binom_U01_f U01, binom_U64_f U64, void* ctx){
    zexp_init_();
    if (clamp01_open(U01(ctx)) < zexp_tail) {
        return zexp_x[ZEXP_N] - log(clamp01_open(U01(ctx)));
    }
    for (;;){
        uint64_t r = U64(ctx);
        int i = (int)(r & 0xFFu);
        double u = (double)(r >> 11) * (1.0/9007199254740992.0);
        double x = zexp_x[i] + u * zexp_w[i];
        double y = clamp01_open(U01(ctx));
        if (y <= zexp_r[i] || y <= exp(-x) / zexp_f[i]) return x;
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
