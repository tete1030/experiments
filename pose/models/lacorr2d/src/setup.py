from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
        "build_ext": BuildExtension
    }
)
