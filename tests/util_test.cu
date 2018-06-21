#include <climits>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <matchbox/util.cuh>

#include <ctime>

namespace matchbox
{
namespace testing
{

MATCHBOX_GLOBAL
void WarpMinTest(const int* __restrict__ values, int* min_values)
{
  const int value = values[threadIdx.x];
  min_values[threadIdx.x] = WarpMin(value, blockDim.x);
}

MATCHBOX_GLOBAL
void BlockMinTest(const int* __restrict__ values, int* min_values)
{
  const int value = values[threadIdx.x];
  min_values[threadIdx.x] = BlockMin(value, threadIdx.x, blockDim.x);
}

MATCHBOX_GLOBAL
void WarpMinIndexTest(const int* __restrict__ values, int* min_values,
    int* min_indices)
{
  int index = threadIdx.x;
  int value = values[threadIdx.x];
  WarpMinIndex(value, index, blockDim.x);
  min_values[threadIdx.x] = value;
  min_indices[threadIdx.x] = index;
}

MATCHBOX_GLOBAL
void BlockMinIndexTest(const int* __restrict__ values, int* min_values,
    int* min_indices)
{
  int index = threadIdx.x;
  int value = values[threadIdx.x];
  BlockMinIndex(value, index, blockDim.x);
  min_values[threadIdx.x] = value;
  min_indices[threadIdx.x] = index;
}

TEST(Util, WarpMin)
{
  thrust::host_vector<int> host_buffer(32);

  for (int j = 0; j < host_buffer.size(); ++j)
  {
    int expected_value = std::numeric_limits<int>::max();
    int expected_index = 0;

    for (int i = 0; i < host_buffer.size(); ++i)
    {
      host_buffer[i] = (j + i) % host_buffer.size() + 7;

      if (host_buffer[i] < expected_value)
      {
        expected_value = host_buffer[i];
        expected_index = i;
      }
    }

    if (expected_index + 1 < host_buffer.size())
    {
      host_buffer[expected_index + 1] = expected_value;
    }

    const int count = host_buffer.size();
    thrust::device_vector<int> found(count);
    thrust::device_vector<int> buffer(host_buffer);
    const int* src = thrust::raw_pointer_cast(buffer.data());
    int* dst = thrust::raw_pointer_cast(found.data());
    CUDA_LAUNCH(WarpMinTest, 1, count, 0, 0, src, dst);

    for (int i = 0; i < count; ++i)
    {
      ASSERT_EQ(expected_value, found[i]);
    }
  }
}

TEST(Util, SubWarpMin)
{
  thrust::host_vector<int> host_buffer(16);

  for (int j = 0; j < host_buffer.size(); ++j)
  {
    int expected_value = std::numeric_limits<int>::max();
    int expected_index = 0;

    for (int i = 0; i < host_buffer.size(); ++i)
    {
      host_buffer[i] = (j + i) % host_buffer.size() + 7;

      if (host_buffer[i] < expected_value)
      {
        expected_value = host_buffer[i];
        expected_index = i;
      }
    }

    if (expected_index + 1 < host_buffer.size())
    {
      host_buffer[expected_index + 1] = expected_value;
    }

    const int count = host_buffer.size();
    thrust::device_vector<int> found(count);
    thrust::device_vector<int> buffer(host_buffer);
    const int* src = thrust::raw_pointer_cast(buffer.data());
    int* dst = thrust::raw_pointer_cast(found.data());
    CUDA_LAUNCH(WarpMinTest, 1, count, 0, 0, src, dst);

    for (int i = 0; i < count; ++i)
    {
      ASSERT_EQ(expected_value, found[i]);
    }
  }
}

TEST(Util, BlockMin)
{
  thrust::host_vector<int> host_buffer(128);

  for (int j = 0; j < host_buffer.size(); ++j)
  {
    int expected_value = std::numeric_limits<int>::max();
    int expected_index = 0;

    for (int i = 0; i < host_buffer.size(); ++i)
    {
      host_buffer[i] = (j + i) % host_buffer.size() + 7;

      if (host_buffer[i] < expected_value)
      {
        expected_value = host_buffer[i];
        expected_index = i;
      }
    }

    if (expected_index + 1 < host_buffer.size())
    {
      host_buffer[expected_index + 1] = expected_value;
    }

    const int count = host_buffer.size();
    thrust::device_vector<int> found(count);
    thrust::device_vector<int> buffer(host_buffer);
    int* dst = thrust::raw_pointer_cast(found.data());
    const int* src = thrust::raw_pointer_cast(buffer.data());
    CUDA_LAUNCH(BlockMinTest, 1, count, 0, 0, src, dst);

    for (int i = 0; i < count; ++i)
    {
      ASSERT_EQ(expected_value, found[i]);
    }
  }
}

TEST(Util, WarpMinIndex)
{
  thrust::host_vector<int> host_buffer(32);

  for (int j = 0; j < host_buffer.size(); ++j)
  {
    int expected_value = std::numeric_limits<int>::max();
    int expected_index = 0;

    for (int i = 0; i < host_buffer.size(); ++i)
    {
      host_buffer[i] = (j + i) % host_buffer.size() + 7;

      if (host_buffer[i] < expected_value)
      {
        expected_value = host_buffer[i];
        expected_index = i;
      }
    }

    if (expected_index + 1 < host_buffer.size())
    {
      host_buffer[expected_index + 1] = expected_value;
    }

    const int count = host_buffer.size();
    thrust::device_vector<int> found_values(count);
    thrust::device_vector<int> found_indices(count);
    thrust::device_vector<int> buffer(host_buffer);
    const int* src = thrust::raw_pointer_cast(buffer.data());
    int* vdst = thrust::raw_pointer_cast(found_values.data());
    int* idst = thrust::raw_pointer_cast(found_indices.data());
    CUDA_LAUNCH(WarpMinIndexTest, 1, count, 0, 0, src, vdst, idst);

    for (int i = 0; i < count; ++i)
    {
      ASSERT_EQ(expected_value, found_values[i]);
      ASSERT_EQ(expected_index, found_indices[i]);

    }
  }
}

TEST(Util, SubWarpMinIndex)
{
  thrust::host_vector<int> host_buffer(16);

  for (int j = 0; j < host_buffer.size(); ++j)
  {
    int expected_value = std::numeric_limits<int>::max();
    int expected_index = 0;

    for (int i = 0; i < host_buffer.size(); ++i)
    {
      host_buffer[i] = (j + i) % host_buffer.size() + 7;

      if (host_buffer[i] < expected_value)
      {
        expected_value = host_buffer[i];
        expected_index = i;
      }
    }

    if (expected_index + 1 < host_buffer.size())
    {
      host_buffer[expected_index + 1] = expected_value;
    }

    const int count = host_buffer.size();
    thrust::device_vector<int> found_values(count);
    thrust::device_vector<int> found_indices(count);
    thrust::device_vector<int> buffer(host_buffer);
    const int* src = thrust::raw_pointer_cast(buffer.data());
    int* vdst = thrust::raw_pointer_cast(found_values.data());
    int* idst = thrust::raw_pointer_cast(found_indices.data());
    CUDA_LAUNCH(WarpMinIndexTest, 1, count, 0, 0, src, vdst, idst);

    for (int i = 0; i < count; ++i)
    {
      ASSERT_EQ(expected_value, found_values[i]);
      ASSERT_EQ(expected_index, found_indices[i]);
    }
  }
}

TEST(Util, BlockMinIndex)
{
  thrust::host_vector<int> host_buffer(128);

  for (int j = 0; j < host_buffer.size(); ++j)
  {
    int expected_value = std::numeric_limits<int>::max();
    int expected_index = 0;

    for (int i = 0; i < host_buffer.size(); ++i)
    {
      host_buffer[i] = (j + i) % host_buffer.size() + 7;

      if (host_buffer[i] < expected_value)
      {
        expected_value = host_buffer[i];
        expected_index = i;
      }
    }

    if (expected_index + 1 < host_buffer.size())
    {
      host_buffer[expected_index + 1] = expected_value;
    }

    const int count = host_buffer.size();
    thrust::device_vector<int> found_values(count);
    thrust::device_vector<int> found_indices(count);
    thrust::device_vector<int> buffer(host_buffer);
    const int* src = thrust::raw_pointer_cast(buffer.data());
    int* vdst = thrust::raw_pointer_cast(found_values.data());
    int* idst = thrust::raw_pointer_cast(found_indices.data());
    CUDA_LAUNCH(BlockMinIndexTest, 1, buffer.size(), 0, 0, src, vdst, idst);

    for (int i = 0; i < count; ++i)
    {
      ASSERT_EQ(expected_value, found_values[i]);
      ASSERT_EQ(expected_index, found_indices[i]);
    }
  }
}

} // namespace testing

} // namespace matchbox