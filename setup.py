from setuptools import setup, Extension
from Cython.Build import cythonize

import numpy
import os
import platform
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

system = platform.system()

if system == "Linux":
    compile_args = ['-fopenmp', '-O3', '-ffast-math', '-march=native', '-fno-wrapv']
    link_args = ['-fopenmp', '-lm']
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"
elif system == "Darwin":  # macOS
    compile_args = ['-O3', '-ffast-math', '-fno-wrapv']
    link_args = ['-lm']
    os.environ["CC"] = "clang"
    os.environ["CXX"] = "clang++"
elif system == "Windows":
    if os.environ.get("CC", "").endswith("gcc"):
        compile_args = ['-O3', '-fopenmp']
    else:
        compile_args = ['/O2', '/openmp']
else:
    log.info(f"System not recognized: {system}")
    sys.exit(1)

common_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

# Define extensions
extensions = [
    Extension(
        name="neural_admixture.src.utils_c.utils",
        sources=["neural_admixture/src/utils_c/utils.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=common_macros,
    ),
    Extension(
        name="neural_admixture.src.utils_c.rsvd",
        sources=["neural_admixture/src/utils_c/rsvd.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=common_macros,
    )
]

setup(
    ext_modules=cythonize(extensions),
    include_package_data=True,
)
