#pragma once

#include <matchbox/device.h>

namespace matchbox
{

MATCHBOX_DEVICE
inline int ShuffleDown(int value, int lane, int width)
{
  return __shfl_down(value, lane, width);
}

template <typename T>
MATCHBOX_DEVICE
inline T WarpMin(T value, int warp_size = 32)
{
  for (int i = warp_size >> 1; i > 0; i >>= 1)
  {
    T temp = ShuffleDown(value, i, warp_size);
    if (temp < value) value = temp;
  }

  return __shfl(value, 0, 32);
}

template <typename T>
MATCHBOX_DEVICE
inline T BlockMin(T value, int thread, int block_size)
{
  MATCHBOX_SHARED T shared[32];
  const int lane = thread % 32;
  const int warp = thread / 32;
  value = WarpMin(value);

  if (lane == 0)
  {
    shared[warp] = value;
  }

  __syncthreads();
  value = shared[lane];
  return WarpMin(value, CeilDiv(block_size, 32));
}

template <typename T>
MATCHBOX_DEVICE
inline void WarpMinIndex(T& value, int& index, int warp_size = 32)
{
  for (int i = warp_size >> 1; i > 0; i >>= 1)
  {
    T temp_value = ShuffleDown(value, i, warp_size);
    int temp_index = ShuffleDown(index, i, warp_size);

    // if (temp_value < value) // TODO: check if can be used instead
    if ((temp_value < value) || (temp_value == value && temp_index < index))
    {
      value = temp_value;
      index = temp_index;
    }
  }

  value = __shfl(value, 0, 32);
  index = __shfl(index, 0, 32);
}

template <typename T>
MATCHBOX_DEVICE
inline void BlockMinIndex(T& value, int& index, int block_size)
{
  MATCHBOX_SHARED T shared_value[32];
  MATCHBOX_SHARED int shared_index[32];
  const int lane = index % 32;
  const int warp = index / 32;

  WarpMinIndex(value, index);

  if (lane == 0)
  {
    shared_value[warp] = value;
    shared_index[warp] = index;
  }

  __syncthreads();

  value = shared_value[lane];
  index = shared_index[lane];
  WarpMinIndex(value, index, CeilDiv(block_size, 32));
}

} // namespace matchbox