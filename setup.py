from pathlib import Path

from setuptools import Extension, setup


ROOT = Path(__file__).parent


def numpy_include():
    import numpy

    return numpy.get_include()


extensions = [
    Extension(
        "categorical_samplers._native",
        sources=[
            str(ROOT / "native" / "python" / "categorical_samplers_native.c"),
            str(ROOT / "native" / "core" / "binom_core.c"),
        ],
        include_dirs=[
            str(ROOT / "native" / "core" / "include"),
            numpy_include(),
        ],
        extra_compile_args=["-O3"],
        language="c",
    )
]


setup(ext_modules=extensions)
