#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t d;
    double *p_sorted;
    size_t *perm;
} cs_multinom_pivot_state;

typedef struct {
    size_t d;
    double *cdf;
} cs_multinom_cdf_state;

typedef struct {
    size_t d;
    double *prob;
    size_t *alias;
} cs_multinom_alias_state;

enum {
    CS_MULTINOM_OK = 0,
    CS_MULTINOM_BAD_PROB = 1,
    CS_MULTINOM_ALLOC_FAILED = 2
};

int cs_multinom_pivot_build(const double *p, size_t d, cs_multinom_pivot_state *state);
void cs_multinom_pivot_free(cs_multinom_pivot_state *state);
void cs_multinom_pivot_draw_one(const cs_multinom_pivot_state *state,
                                uint64_t K,
                                uint64_t seed,
                                int64_t *out);
void cs_multinom_pivot_draw_batch(const cs_multinom_pivot_state *state,
                                  uint64_t K,
                                  size_t m,
                                  uint64_t seed,
                                  int64_t *out_row_major);

int cs_multinom_cdf_build(const double *p, size_t d, cs_multinom_cdf_state *state);
void cs_multinom_cdf_free(cs_multinom_cdf_state *state);
void cs_multinom_cdf_draw_batch(const cs_multinom_cdf_state *state,
                                uint64_t K,
                                size_t m,
                                uint64_t seed,
                                int64_t *out_row_major);

int cs_multinom_alias_build(const double *p, size_t d, cs_multinom_alias_state *state);
void cs_multinom_alias_free(cs_multinom_alias_state *state);
void cs_multinom_alias_draw_batch(const cs_multinom_alias_state *state,
                                  uint64_t K,
                                  size_t m,
                                  uint64_t seed,
                                  int64_t *out_row_major);

#ifdef __cplusplus
}
#endif
