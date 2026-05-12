#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* RNG callbacks supplied by the caller (your MEX):
   - U01(ctx): return U in (0,1), not including 0 or 1
   - U64(ctx): return 64 uniformly random bits (for ziggurat fast path)
*/
typedef double   (*binom_U01_f)(void* ctx);
typedef uint64_t (*binom_U64_f)(void* ctx);

/* Exact Binomial via center–out inversion around the mode. */
size_t binom_centerout_core(size_t n, double p,
                            binom_U01_f U01,
                            void* ctx);

/* Exact Binomial via waiting-times (Exp(1) inter-arrivals).
   Uses Ziggurat Exp(1) for the Exp deviates (log-free in the hot path),
   and falls back to -log(U) only in rare slow branches.
*/
size_t binom_wait2_core(size_t n, double p,
                        binom_U01_f U01,
                        binom_U64_f U64,
                        void* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif
