#pragma once

#include <cuda_runtime.h>
#include <matchbox/exception.h>

#define MATCHBOX_GLOBAL      __global__
#define MATCHBOX_SHARED      __shared__
#define MATCHBOX_HOST        __host__
#define MATCHBOX_DEVICE      __device__
#define MATCHBOX_HOST_DEVICE __host__ __device__

#define CUDA_ASSERT(cmd) {                                                     \
  const cudaError_t code = cmd;                                                \
  MATCHBOX_ASSERT_MSG(code == cudaSuccess, GetCudaErroString(code));           \
}

#ifdef NDEBUG

#define CUDA_DEBUG(cmd) cmd

#define CUDA_LAUNCH(kernel, grids, blocks, shared, stream, ...)                \
  kernel<<<grids, blocks, shared, stream>>>(__VA_ARGS__)

#else

#define CUDA_DEBUG CUDA_ASSERT

#define CUDA_LAUNCH(kernel, grids, blocks, shared, stream, ...) {              \
  kernel<<<grids, blocks, shared, stream>>>(__VA_ARGS__);                      \
  CUDA_DEBUG(cudaDeviceSynchronize());                                         \
  CUDA_DEBUG(cudaGetLastError());                                              \
}

#endif

#ifdef __CUDA_ARCH__
#define DEVICE_RETURN(value) return value
#else
#define DEVICE_RETURN(value)
#endif

namespace matchbox
{

MATCHBOX_HOST_DEVICE
inline int CeilDiv(int a, int b)
{
  return (a + b - 1) / b;
}

MATCHBOX_HOST_DEVICE
inline int GetGrids(int threads, int blocks)
{
  return CeilDiv(threads, blocks);
}

MATCHBOX_HOST_DEVICE
inline dim3 GetGrids(dim3 threads, dim3 blocks)
{
  dim3 result;
  result.x = GetGrids(threads.x, blocks.x);
  result.y = GetGrids(threads.y, blocks.y);
  result.z = GetGrids(threads.z, blocks.z);
  return result;
}

#ifdef __CUDA_ARCH__

inline const char* GetCudaErroString(cudaError_t code)
{
  return cudaGetErrorString(code);
}

#else

inline std::string GetCudaErroString(cudaError_t code)
{
  const std::string error = cudaGetErrorString(code);
  return error + " [cuda error " + std::to_string(code) + "]";
}

#endif

} // namespace matchbox