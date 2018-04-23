import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(os.path.abspath(__file__))

sources = ['src/ae_loss.c']
headers = ['src/ae_loss.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources = ['src/ae_loss.c']
    headers = ['src/ae_loss.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

extra_objects = ['src/ae_loss_kernel.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '.lib',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    os.system("cd src;nvcc ae_loss_kernel.cu -c -o ae_loss_kernel.o -x cu -Xcompiler -fPIC -arch=sm_52")
    ffi.build()