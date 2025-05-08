from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define extensions
cython_ext = Extension(
    name="neural_admixture.src.utils_c.utils",
    sources=["neural_admixture/src/utils_c/utils.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-fopenmp', '-O3', '-ffast-math', '-march=native', '-fno-wrapv'],
    extra_link_args=['-fopenmp', '-lm'],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
)

setup(
    ext_modules=cythonize([cython_ext]),
)