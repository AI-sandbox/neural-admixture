from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

common_compile_args = ['-fopenmp', '-O3', '-ffast-math', '-march=native', '-fno-wrapv']
common_link_args = ['-fopenmp', '-lm']
common_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

# Define extensions
extensions = [
    Extension(
        name="neural_admixture.src.utils_c.utils",
        sources=["neural_admixture/src/utils_c/utils.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
        define_macros=common_macros,
    ),
    Extension(
        name="neural_admixture.src.utils_c.rsvd",
        sources=["neural_admixture/src/utils_c/rsvd.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
        define_macros=common_macros,
    )
]

setup(
    ext_modules=cythonize(extensions),
)
