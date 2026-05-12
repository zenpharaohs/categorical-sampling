from setuptools import Extension, setup


def numpy_include():
    import numpy

    return numpy.get_include()


extensions = [
    Extension(
        "categorical_samplers._native",
        sources=[
            "native/python/categorical_samplers_native.c",
            "native/core/binom_core.c",
        ],
        include_dirs=[
            "native/core/include",
            numpy_include(),
        ],
        extra_compile_args=["-O3"],
        language="c",
    )
]


setup(ext_modules=extensions)
