#include <matchbox/disparity_computer.h>
#include <matchbox/aggregate_cost.h>
#include <matchbox/device.h>
#include <matchbox/exception.h>
#include <matchbox/image.h>
#include <matchbox/util.cuh>

namespace matchbox
{

template <typename T>
MATCHBOX_DEVICE
inline T WarpMinIndex2(T value, int index)
{
  for (int i = 16; i > 0; i >>= 1)
  {
    T temp_value   = __shfl_down(value, i, 32);
    int temp_index = __shfl_down(index, i, 32);

    if (temp_value < value)
    {
      value = temp_value;
      index = temp_index;
    }
  }

  return __shfl(index, 0, 32);
}

template <int MAX_DISP, int PATHS>
MATCHBOX_GLOBAL
void ComputeKernel(const uint8_t* __restrict__ costs, uint8_t* disparities,
    float uniqueness)
{
  const int w = gridDim.x;
  const int h = 2 * gridDim.y;
  const int x = blockIdx.x;
  const int y = 2 * blockIdx.y + threadIdx.y;

  uint32_t nc0 = 0;
  uint32_t nc1 = 0;
  const uint32_t zero = 0;

  const int step = h * w * MAX_DISP;
  const int offset = y * w * MAX_DISP + x * MAX_DISP;
  const uint32_t* cc = reinterpret_cast<const uint32_t*>(costs);

  for (int p = 0; p < PATHS; ++p)
  {
    const uint32_t c = cc[((p * step + offset) >> 2) + threadIdx.x];
    uint32_t cc0 = __byte_perm(c, zero, 0x4140);
    uint32_t cc1 = __byte_perm(c, zero, 0x4342);
    nc0 = __vadd2(nc0, cc0);
    nc1 = __vadd2(nc1, cc1);
  }

  const uint32_t mask = __vcmpleu2(nc0, nc1);
  uint32_t a = (mask & nc0) | ((mask ^ 0xFFFFFFFF) & nc1);
  uint32_t b = (mask & 0x00010000) | ((mask ^ 0xFFFFFFFF) & 0x00030002);

  if ((a & 0xFFFF) > (a >> 16))
  {
    a >>= 16;
    b >>= 16;
  }

  const uint32_t cost = a & 0xFFFF;
  const int dd = 4 * threadIdx.x + (b & 0xFFFF);
  int d = threadIdx.x;

  // d = WarpMinIndex2(cost, d);
  // if (threadIdx.x == d) disparities[y * w + x] = dd;

  uint32_t temp_cost = cost;
  int temp_d = d;

  d = WarpMinIndex2(cost, d);

  if (threadIdx.x == d)
  {
    temp_cost = 255;
  }

  temp_d = WarpMinIndex2(temp_cost, temp_d);

  if (threadIdx.x == d)
  {
    disparities[y * w + x] = (uniqueness * temp_cost < cost &&
        abs(temp_d - d) > 1) ? 0 : dd;
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void ComputeInvertedKernel(const uint8_t* __restrict__ costs,
    uint8_t* disparities, int paths, float uniqueness)
{
  const int w = gridDim.x;
  const int h = gridDim.y;
  const int x = blockIdx.x;
  const int y = blockIdx.y;
  int d = threadIdx.x;
  uint32_t cost = 0;

  const int step = h * w * MAX_DISP;
  const int offset = y * w * MAX_DISP + (x + d) * MAX_DISP + d;

  if (x + d < w)
  {
    for (int p = 0; p < paths; ++p)
    {
      cost += costs[p * step + offset];
    }
  }
  else
  {
    cost = paths * (64 + 100);
  }

  uint32_t temp_cost = cost;
  int temp_d = d;

  BlockMinIndex(cost, d, MAX_DISP);

  __syncthreads();

  if (threadIdx.x == d)
  {
    temp_cost = 255;
  }

  BlockMinIndex(temp_cost, temp_d, MAX_DISP);

  if (threadIdx.x == 0)
  {
    disparities[y * w + x] = (uniqueness * temp_cost < cost &&
        abs(temp_d - d) > 1) ? 0 : d;
  }
}

DisparityComputer::DisparityComputer(
    std::shared_ptr<const AggregateCost> cost) :
  cost_(cost),
  uniqueness_(0.95),
  inverted_(false)
{
}

std::shared_ptr<const AggregateCost> DisparityComputer::GetCost() const
{
  return cost_;
}

float DisparityComputer::GetUniqueness() const
{
  return uniqueness_;
}

void DisparityComputer::SetUniqueness(float uniqueness)
{
  MATCHBOX_DEBUG(uniqueness > 0.0 && uniqueness <= 1.0);
  uniqueness_ = uniqueness;
}

bool DisparityComputer::IsInverted() const
{
  return inverted_;
}

void DisparityComputer::SetInverted(bool inverted)
{
  inverted_ = inverted;
}

void DisparityComputer::Compute(Image& image) const
{
  MATCHBOX_ASSERT(cost_->GetDepth() == 128);

  const int w = cost_->GetWidth();
  const int h = cost_->GetHeight();
  const int d = cost_->GetDepth();
  const int p = cost_->GetPaths();

  const dim3 grids(w, h);
  const dim3 blocks(d);

  image.SetSize(w, h);
  uint8_t* dst = image.GetData();
  const uint8_t* src = cost_->GetData();
  float u = uniqueness_;

  if (inverted_)
  {
    CUDA_LAUNCH(ComputeInvertedKernel<128>, grids, blocks, 0, 0, src, dst, p, u);
  }
  else
  {
    const dim3 grids(w, h / 2);
    const dim3 blocks(d / 4, 2);

    switch (p)
    {
      case 1 : CUDA_LAUNCH((ComputeKernel<128, 1>), grids, blocks, 0, 0, src, dst, u); break;
      case 2 : CUDA_LAUNCH((ComputeKernel<128, 2>), grids, blocks, 0, 0, src, dst, u); break;
      case 4 : CUDA_LAUNCH((ComputeKernel<128, 4>), grids, blocks, 0, 0, src, dst, u); break;
      case 8 : CUDA_LAUNCH((ComputeKernel<128, 8>), grids, blocks, 0, 0, src, dst, u); break;
    }
  }
}

} // namespace matchbox