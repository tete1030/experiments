#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_NUM_THREADS 1024
#define CUDA_KERNEL_LOOP(i, n) \
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (int64_t)blockDim.x * gridDim.x)

inline int64_t GET_BLOCKS(const int64_t N)
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