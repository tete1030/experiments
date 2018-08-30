from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import setuptools
import os
import sys

setup(
    name="displace_cuda",
    ext_modules=[
        CUDAExtension(
            name="displace_cuda",
            sources=[
                "displace.cpp",
                "displace_kernel.cu"
            ],
            extra_compile_args={"cxx": [],
                                "nvcc": ["-gencode=arch=compute_61,code=sm_61", "--ptxas-options=-v"]})
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
