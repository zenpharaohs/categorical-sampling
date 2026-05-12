#pragma once
#include <float.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t state;
} cs_rng64_t;

static inline uint64_t cs_splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static inline void cs_rng64_seed(cs_rng64_t *rng, uint64_t seed) {
    if (seed == 0) {
        seed = UINT64_C(0xDEADBEEFCAFEBABE);
    }
    rng->state = cs_splitmix64_next(&seed);
    if (rng->state == 0) {
        rng->state = UINT64_C(0xA5A5A5A5A5A5A5A5);
    }
}

static inline uint64_t cs_rng64_u64(cs_rng64_t *rng) {
    return cs_splitmix64_next(&rng->state);
}

static inline double cs_rng64_u01_open(cs_rng64_t *rng) {
    const uint64_t r = cs_rng64_u64(rng);
    const uint64_t x = (r >> 11) | UINT64_C(1);
    double u = (double)x * (1.0 / 9007199254740992.0);
    if (!(u > 0.0)) {
        u = DBL_MIN;
    }
    if (u >= 1.0) {
        u = 1.0 - DBL_MIN;
    }
    return u;
}

#ifdef __cplusplus
}
#endif
