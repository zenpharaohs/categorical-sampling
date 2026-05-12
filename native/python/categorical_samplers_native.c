#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_20_API_VERSION
#include <numpy/arrayobject.h>

#include "binom_core.h"
#include "multinom_core.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint64_t state;
} rng_ctx_t;

static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double cb_u01(void *ctx) {
    rng_ctx_t *rng = (rng_ctx_t *)ctx;
    const uint64_t r = splitmix64_next(&rng->state);
    const uint64_t m = (r >> 11) | 1ULL;
    return (double)m * (1.0 / 9007199254740992.0);
}

static uint64_t cb_u64(void *ctx) {
    rng_ctx_t *rng = (rng_ctx_t *)ctx;
    return splitmix64_next(&rng->state);
}

static uint64_t seed_from_none(void) {
    uint64_t seed = (uint64_t)time(NULL);
    seed ^= (uint64_t)(uintptr_t)&seed;
    seed ^= (uint64_t)clock() << 32;
    return seed;
}

static int read_seed(PyObject *seed_obj, uint64_t *seed_out) {
    if (seed_obj == NULL || seed_obj == Py_None) {
        *seed_out = seed_from_none();
        return 1;
    }
    unsigned long long raw = PyLong_AsUnsignedLongLongMask(seed_obj);
    if (PyErr_Occurred()) {
        return 0;
    }
    *seed_out = (uint64_t)raw;
    return 1;
}

static PyObject *native_available(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    Py_RETURN_TRUE;
}

static PyObject *native_binomial(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"n", "p", "size", "seed", "method", NULL};

    unsigned long long n_raw = 0;
    double p = 0.0;
    Py_ssize_t size = 1;
    PyObject *seed_obj = Py_None;
    const char *method = "auto";

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "Kd|nOs:binomial",
            kwlist,
            &n_raw,
            &p,
            &size,
            &seed_obj,
            &method)) {
        return NULL;
    }

    if (size < 0) {
        PyErr_SetString(PyExc_ValueError, "size must be nonnegative");
        return NULL;
    }
    if (!(p >= 0.0 && p <= 1.0)) {
        PyErr_SetString(PyExc_ValueError, "p must be in [0, 1]");
        return NULL;
    }
    if (n_raw > (unsigned long long)LLONG_MAX) {
        PyErr_SetString(PyExc_OverflowError, "n is too large for int64 output");
        return NULL;
    }

    uint64_t seed = 0;
    if (!read_seed(seed_obj, &seed)) {
        return NULL;
    }

    int use_wait2 = 0;
    if (strcmp(method, "wait2") == 0) {
        use_wait2 = 1;
    } else if (strcmp(method, "auto") == 0) {
        double peff = (p <= 0.5) ? p : (1.0 - p);
        double mean = (double)n_raw * peff;
        use_wait2 = (peff <= 0.01 || mean <= 64.0);
    } else if (strcmp(method, "centerout") == 0 || strcmp(method, "btrd") == 0) {
        use_wait2 = 0;
    } else {
        PyErr_SetString(PyExc_ValueError, "method must be 'auto', 'centerout', 'wait2', or 'btrd'");
        return NULL;
    }

    npy_intp dims[1] = {(npy_intp)size};
    PyObject *array = PyArray_SimpleNew(1, dims, NPY_INT64);
    if (array == NULL) {
        return NULL;
    }
    int64_t *out = (int64_t *)PyArray_DATA((PyArrayObject *)array);

    const size_t n = (size_t)n_raw;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < size; ++i) {
        rng_ctx_t ctx;
        ctx.state = seed ^ (0xC6A4A7935BD1E995ULL * ((uint64_t)i + 1ULL));
        size_t value = use_wait2
            ? binom_wait2_core(n, p, cb_u01, cb_u64, &ctx)
            : binom_centerout_core(n, p, cb_u01, &ctx);
        out[i] = (int64_t)value;
    }
    Py_END_ALLOW_THREADS

    return array;
}

static PyObject *native_multinomial(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"K", "p", "size", "seed", "method", NULL};

    unsigned long long K_raw = 0;
    PyObject *p_obj = NULL;
    Py_ssize_t size = 1;
    PyObject *seed_obj = Py_None;
    const char *method = "auto";

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "KO|nOs:multinomial",
            kwlist,
            &K_raw,
            &p_obj,
            &size,
            &seed_obj,
            &method)) {
        return NULL;
    }
    if (size < 0) {
        PyErr_SetString(PyExc_ValueError, "size must be nonnegative");
        return NULL;
    }
    if (strcmp(method, "auto") != 0 && strcmp(method, "pivot") != 0 && strcmp(method, "cascade") != 0) {
        PyErr_SetString(PyExc_ValueError, "method must be 'auto', 'pivot', or 'cascade'");
        return NULL;
    }
    if (K_raw > (unsigned long long)LLONG_MAX) {
        PyErr_SetString(PyExc_OverflowError, "K is too large for int64 output");
        return NULL;
    }

    PyArrayObject *p_arr = (PyArrayObject *)PyArray_FROM_OTF(p_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (p_arr == NULL) {
        return NULL;
    }
    if (PyArray_NDIM(p_arr) != 1) {
        Py_DECREF(p_arr);
        PyErr_SetString(PyExc_ValueError, "p must be a one-dimensional probability vector");
        return NULL;
    }
    npy_intp d_np = PyArray_DIM(p_arr, 0);
    if (d_np <= 0 || d_np > PY_SSIZE_T_MAX) {
        Py_DECREF(p_arr);
        PyErr_SetString(PyExc_ValueError, "p must contain at least one category");
        return NULL;
    }

    uint64_t seed = 0;
    if (!read_seed(seed_obj, &seed)) {
        Py_DECREF(p_arr);
        return NULL;
    }

    cs_multinom_pivot_state state;
    int rc = cs_multinom_pivot_build((const double *)PyArray_DATA(p_arr), (size_t)d_np, &state);
    Py_DECREF(p_arr);
    if (rc == CS_MULTINOM_BAD_PROB) {
        PyErr_SetString(PyExc_ValueError, "p must contain finite nonnegative values with positive total mass");
        return NULL;
    }
    if (rc == CS_MULTINOM_ALLOC_FAILED) {
        return PyErr_NoMemory();
    }
    if (rc != CS_MULTINOM_OK) {
        PyErr_SetString(PyExc_RuntimeError, "failed to build multinomial pivot state");
        return NULL;
    }

    npy_intp dims[2] = {(npy_intp)size, d_np};
    PyObject *array = PyArray_SimpleNew(2, dims, NPY_INT64);
    if (array == NULL) {
        cs_multinom_pivot_free(&state);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    cs_multinom_pivot_draw_batch(
        &state,
        (uint64_t)K_raw,
        (size_t)size,
        seed,
        (int64_t *)PyArray_DATA((PyArrayObject *)array));
    Py_END_ALLOW_THREADS

    cs_multinom_pivot_free(&state);
    return array;
}

static PyMethodDef methods[] = {
    {"native_available", native_available, METH_NOARGS, "Return True when the native extension is loaded."},
    {"binomial", (PyCFunction)native_binomial, METH_VARARGS | METH_KEYWORDS, "Draw native binomial variates."},
    {"multinomial", (PyCFunction)native_multinomial, METH_VARARGS | METH_KEYWORDS, "Draw native multinomial count vectors."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Native categorical sampler backend.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__native(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
