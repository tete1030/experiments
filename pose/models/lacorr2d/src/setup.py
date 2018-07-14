from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, library_paths, include_paths
import setuptools
import os
import sys

class MyBuildExtension(BuildExtension):
    def build_extensions(self):
        if "DEBUG" in os.environ and os.environ["DEBUG"] == "1":
            self._add_compile_arg("-DDEBUG")
        if "FLOAT_ONLY" in os.environ and os.environ["FLOAT_ONLY"] == "1":
            self._add_compile_arg("-DFLOAT_ONLY=1")
        super(MyBuildExtension, self).build_extensions()

    def _add_compile_arg(self, arg):
        for extension in self.extensions:
            if isinstance(extension.extra_compile_args, dict):
                for args in extension.extra_compile_args.values():
                    args.append(arg)
            else:
                extension.extra_compile_args.append(arg)

setup(
    name="lacorr2d_cuda",
    ext_modules=[
        CUDAExtension(
            name="lacorr2d_cuda",
            sources=[
                "lacorr2d.cpp",
                "lacorr2d_kernel.cu"
            ],
            extra_compile_args={"cxx": [],
                                "nvcc": ["-gencode=arch=compute_61,code=sm_61"]})
    ],
    cmdclass={
        "build_ext": MyBuildExtension
    }
)
