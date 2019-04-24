from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import setuptools
import os
import sys

print("Warning: this is for experiment code with commit id older than 12e66d178c0c9110055ec23dc4c164f030f96788", file=sys.stderr)

CC=os.environ["CUDA_CC"] if "CUDA_CC" in os.environ else "61"

setup(
    name="displace_cuda",
    ext_modules=[
        CUDAExtension(
            name="displace_cuda",
            sources=[
                "displace.cpp",
                "displace_kernel.cu",
                "displace_pos_kernel.cu",
                "displace_gaus_kernel.cu"
            ],
            extra_compile_args={"cxx": [],
                                "nvcc": ["-gencode=arch=compute_{},code=sm_{}".format(CC, CC), "--ptxas-options=-v"]})
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
