
// sampler_core.h
// Minimal helpers: RNG, OpenMP guard, and common utilities.
// Public domain / MIT-style helper header.

#ifndef SAMPLER_CORE_H
#define SAMPLER_CORE_H

#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "mex.h"

// -------- OpenMP guard --------
#ifdef _OPENMP
  #include <omp.h>
#else
  // Fallback stubs
  static inline int omp_get_max_threads(void){ return 1; }
  static inline int omp_get_thread_num(void){ return 0; }
  static inline void omp_set_num_threads(int n) {(void)n;}
#endif

// -------- SplitMix64 (for seeding) --------
static inline uint64_t splitmix64_next(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// -------- xorshift64* --------
typedef struct {
    uint64_t s;
} rng64_t;

static inline void rng64_seed(rng64_t *r, uint64_t seed) {
    if (seed == 0) seed = 0xDEADBEEFCAFEBABEULL;
    // Scramble via splitmix64
    r->s = splitmix64_next(&seed);
    if (r->s == 0) r->s = 0xA5A5A5A5ULL;
}

static inline uint64_t rng64_next_u64(rng64_t *r) {
    uint64_t x = r->s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    r->s = x;
    return x * 2685821657736338717ULL;
}

// uniform in [0,1)
static inline double rng64_uniform01(rng64_t *r){
    // take top 53 bits -> double in [0,1)
    return (rng64_next_u64(r) >> 11) * (1.0/9007199254740992.0); // 2^53
}

// Geometric(p) with support {1,2,...}; returns as int
static inline int sample_geometric(rng64_t *r, double p){
    if (p <= 0.0) return INT_MAX/2;
    if (p >= 1.0) return 1;
    double u = rng64_uniform01(r);
    // ceil(log(1-u)/log(1-p)) but numerically stable:
    return (int)floor(log(1.0 - u) / log(1.0 - p)) + 1;
}

// Binomial via "centered inversion" from the mode (exact).
// Expected O(sqrt(n p (1-p))) steps.
static inline int sample_binom_center_inversion(rng64_t *r, int n, double p){
    if (p <= 0.0) return 0;
    if (p >= 1.0) return n;
    // symmetry: use p <= 0.5
    int flipped = 0;
    if (p > 0.5){
        p = 1.0 - p;
        flipped = 1;
    }
    // handle tiny/huge quickly
    if (n <= 0) return 0;
    // Mode and base pmf at mode
    int m0 = (int)floor((n + 1) * p);
    // Compute pmf(m0) using log to avoid overflow:
    // pmf(k) = C(n,k) p^k (1-p)^(n-k)
    // log C(n,k) via lgamma
    double logpmf_m0 = lgamma(n + 1.0) - lgamma(m0 + 1.0) - lgamma(n - m0 + 1.0)
                     + m0 * log(p) + (n - m0) * log(1.0 - p);
    double pmf_m0 = exp(logpmf_m0);

    double u = rng64_uniform01(r);
    // Expand cdf from mode outward, alternating left/right
    double csum = pmf_m0;
    if (u <= csum){
        return flipped ? (n - m0) : m0;
    }

    double left = pmf_m0;
    double right = pmf_m0;
    int kL = m0 - 1;
    int kR = m0 + 1;
    while (kL >= 0 || kR <= n){
        if (kR <= n){
            // right ratio: pmf(kR) = pmf(kR-1) * (n - (kR-1)) / kR * p/(1-p)
            right *= ((double)(n - (kR - 1)) / (double)kR) * (p / (1.0 - p));
            csum += right;
            if (u <= csum){
                int ans = kR;
                return flipped ? (n - ans) : ans;
            }
            kR++;
        }
        if (kL >= 0){
            // left ratio: pmf(kL) = pmf(kL+1) * (kL+1)/(n - kL) * (1-p)/p
            left *= ((double)(kL + 1) / (double)(n - kL)) * ((1.0 - p)/p);
            csum += left;
            if (u <= csum){
                int ans = kL;
                return flipped ? (n - ans) : ans;
            }
            kL--;
        }
    }
    // Fallback (due to numerical issues): clamp
    int ans = (int)floor((n+1)*p);
    return flipped ? (n - ans) : ans;
}

// Binomial via waiting-times (efficient for small p).
static inline int sample_binom_wait(rng64_t *r, int n, double p){
    if (p <= 0.0) return 0;
    if (p >= 1.0) return n;
    // Use symmetry to keep p <= 0.5
    int flipped = 0;
    if (p > 0.5){
        p = 1.0 - p;
        flipped = 1;
    }
    int t = 0, k = 0;
    while (1){
        int g = sample_geometric(r, p); // failures before next success + 1
        t += g;
        if (t > n) break;
        k++;
    }
    return flipped ? (n - k) : k;
}

// Simple prefix-scan categorical draw (binary search).
static inline int sample_categorical(rng64_t *r, const double *cdf, int K){
    double u = rng64_uniform01(r);
    // binary search
    int lo = 0, hi = K - 1;
    while (lo < hi){
        int mid = (lo + hi) >> 1;
        if (u <= cdf[mid]) hi = mid; else lo = mid + 1;
    }
    return lo; // 0..K-1
}

#endif // SAMPLER_CORE_H
