#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_NUM_THREADS 1024
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

inline int GET_BLOCKS(const int N)
{
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}