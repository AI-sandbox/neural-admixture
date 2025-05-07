from setuptools import setup, Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension as TorchBuildExtension
from torch.utils.cpp_extension import CUDAExtension
import os

# CUDA build flags
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;9.0"
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

class CustomBuildExt(TorchBuildExtension):
    def finalize_options(self):
        super().finalize_options()
        import numpy
        if not hasattr(self, 'include_dirs'):
            self.include_dirs = []
        self.include_dirs.append(numpy.get_include())

# Define extensions
cython_ext = Extension(
    name="neural_admixture.src.utils_c.utils",
    sources=["neural_admixture/src/utils_c/utils.pyx"],
    extra_compile_args=['-fopenmp', '-O3', '-ffast-math', '-march=native', '-fno-wrapv'],
    extra_link_args=['-fopenmp', '-lm'],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
)

cuda_ext = CUDAExtension(
    name="neural_admixture.src.utils_c.pack2bit",
    sources=["neural_admixture/src/utils_c/pack2bit.cu"],
    # You can add nvcc specific compile args here if needed, e.g.:
    # extra_compile_args={'cxx': [], 'nvcc': ['-O3']}
    # TORCH_CUDA_ARCH_LIST environment variable is generally preferred for architecture flags.
)

setup(
    ext_modules=cythonize([cython_ext]) + [cuda_ext],
    cmdclass={'build_ext': CustomBuildExt}
)