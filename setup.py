import numpy
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = "9.0"

from setuptools import setup, Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

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

setup(
    ext_modules=[
        CUDAExtension(
            name='neural_admixture.src.utils_c.pack2bit',
            sources=['neural_admixture/src/utils_c/pack2bit.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)