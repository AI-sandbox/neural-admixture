from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="neural_admixture.src.utils_c.utils",
        sources=["neural_admixture/src/utils_c/utils.pyx"],
        extra_compile_args=['-fopenmp', '-O3', '-ffast-math', '-march=native', '-fno-wrapv'],
        extra_link_args=['-fopenmp', '-lm'],
        include_dirs=[numpy.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    )
]

setup(
    ext_modules=cythonize(extensions),
)
