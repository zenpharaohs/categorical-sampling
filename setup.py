import os
import sys

from setuptools import Extension, setup


def numpy_include():
    import numpy

    return numpy.get_include()


def openmp_flags():
    if os.environ.get("CATEGORICAL_SAMPLERS_NO_OPENMP"):
        return [], [], []
    if sys.platform == "win32":
        return ["/openmp"], [], []
    if sys.platform == "darwin":
        include_dirs = []
        link_args = ["-lomp"]
        for prefix in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
            if os.path.exists(os.path.join(prefix, "include", "omp.h")):
                include_dirs.append(os.path.join(prefix, "include"))
                link_args.insert(0, f"-L{os.path.join(prefix, 'lib')}")
                break
        return ["-Xpreprocessor", "-fopenmp"], link_args, include_dirs
    return ["-fopenmp"], ["-fopenmp"], []


omp_compile, omp_link, omp_include = openmp_flags()


extensions = [
    Extension(
        "categorical_samplers._native",
        sources=[
            "native/python/categorical_samplers_native.c",
            "native/core/binom_core.c",
            "native/core/multinom_core.c",
        ],
        include_dirs=[
            "native/core/include",
            numpy_include(),
            *omp_include,
        ],
        extra_compile_args=["-O3", *omp_compile],
        extra_link_args=[*omp_link],
        language="c",
    )
]


setup(ext_modules=extensions)
