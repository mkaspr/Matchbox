#include <matchbox/disparity_computer.h>
#include <matchbox/aggregate_cost.h>
#include <matchbox/device.h>
#include <matchbox/exception.h>
#include <matchbox/image.h>
#include <matchbox/util.cuh>

namespace matchbox
{

MATCHBOX_DEVICE
inline void WarpMinIndex2(uint32_t& value, uint32_t& index)
{
  if ((value & 0xFFFF) > (value >> 16))
  {
    value = (value << 16) | (value >> 16);
    index = (index << 16) | (index >> 16);
  }

  for (int i = 16; i > 0; i >>= 1)
  {
    const uint32_t temp_value = __shfl_down(value, i, 32);
    const uint32_t temp_index = __shfl_down(index, i, 32);
    const uint32_t mask = __vcmpleu2(value, temp_value);
    value = (mask & value) | ((mask ^ 0xFFFFFFFF) & temp_value);
    index = (mask & index) | ((mask ^ 0xFFFFFFFF) & temp_index);
  }

  value = __shfl(value, 0, 32);
  index = __shfl(index, 0, 32);
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void ComputeKernel(const uint16_t* __restrict__ costs, uint8_t* disparities,
    float uniqueness)
{
  const int w = gridDim.x;
  const int x = blockIdx.x;
  const int y = 2 * blockIdx.y + threadIdx.y;

  const int offset = y * w * MAX_DISP + x * MAX_DISP;
  const uint32_t* cc = reinterpret_cast<const uint32_t*>(costs);
  const uint32_t nc0 = cc[(offset >> 1) + 2 * threadIdx.x + 0];
  const uint32_t nc1 = cc[(offset >> 1) + 2 * threadIdx.x + 1];
  const uint32_t ni0 = ((4 * threadIdx.x + 1) << 16) | (4 * threadIdx.x + 0);
  const uint32_t ni1 = ((4 * threadIdx.x + 3) << 16) | (4 * threadIdx.x + 2);

  const uint32_t mask = __vcmpleu2(nc0, nc1);
  uint32_t value = (mask & nc0) | ((mask ^ 0xFFFFFFFF) & nc1);
  uint32_t index = (mask & ni0) | ((mask ^ 0xFFFFFFFF) & ni1);
  WarpMinIndex2(value, index);

  const uint16_t cost = value & 0xFFFF;
  const uint16_t temp_cost = value >> 16;

  const uint16_t d = index & 0xFFFF;
  const uint16_t temp_d = index >> 16;

  if (threadIdx.x == 0)
  {
    disparities[y * w + x] = (uniqueness * temp_cost < cost &&
        abs(temp_d - d) > 1) ? 0 : d;
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void ComputeInvertedKernel(const uint16_t* __restrict__ costs,
    uint8_t* disparities, int paths, float uniqueness)
{
  const int w = gridDim.x;
  const int x = blockIdx.x;
  const int y = blockIdx.y;
  int d = threadIdx.x;
  uint32_t cost = 0;

  const int offset = y * w * MAX_DISP + (x + d) * MAX_DISP + d;

  if (x + d < w)
  {
    cost = costs[offset];
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

  const dim3 grids(w, h);
  const dim3 blocks(d);

  image.SetSize(w, h);
  uint8_t* dst = image.GetData();
  const uint16_t* src = cost_->GetData();
  float u = uniqueness_;

  if (inverted_)
  {
    CUDA_LAUNCH(ComputeInvertedKernel<128>, grids, blocks, 0, 0, src, dst, 8, u);
  }
  else
  {
    const dim3 grids(w, h / 2);
    const dim3 blocks(d / 4, 2);
    CUDA_LAUNCH(ComputeKernel<128>, grids, blocks, 0, 0, src, dst, u);
  }
}

} // namespace matchbox