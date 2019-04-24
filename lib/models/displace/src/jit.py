from torch.utils.cpp_extension import load
import os

def get_module(cuda_cc=None, cxx=None):
    if cuda_cc is not None:
        CC = cuda_cc
    else:
        CC = os.environ["CUDA_CC"] if "CUDA_CC" in os.environ else "61"

    sources = [
        "displace.cpp",
        "displace_kernel.cu",
        "displace_pos_kernel.cu",
        "displace_gaus_kernel.cu"
    ]

    cur_dir = os.path.abspath(os.path.dirname(__file__))

    for i in range(len(sources)):
        sources[i] = os.path.join(cur_dir, sources[i])

    if cxx is not None:
        ori_cxx_env = os.getenv("CXX")
        os.environ["CXX"] = cxx

    displace_cuda = load(
        name="displace_cuda",
        sources=sources,
        extra_cuda_cflags=["-gencode=arch=compute_{},code=sm_{}".format(CC, CC), "--ptxas-options=-v"])

    if cxx is not None and ori_cxx_env is not None:
        os.environ["CXX"] = ori_cxx_env

    return displace_cuda