#include <matchbox/disparity_computer.h>
#include <matchbox/aggregate_cost.h>
#include <matchbox/device.h>
#include <matchbox/exception.h>
#include <matchbox/image.h>
#include <matchbox/util.cuh>

namespace matchbox
{

template <int MAX_DISP>
MATCHBOX_GLOBAL
void ComputeKernel(const uint16_t* __restrict__ costs, uint8_t* disparities,
    int paths, float uniqueness)
{
  const int w = gridDim.x;
  const int h = gridDim.y;
  const int x = blockIdx.x;
  const int y = blockIdx.y;
  int d0 = 2 * threadIdx.x + 0;
  int d1 = 2 * threadIdx.x + 1;
  uint16_t cost0 = 0;
  uint16_t cost1 = 0;

  const int step = h * w * MAX_DISP;
  const int offset = y * w * MAX_DISP + x * MAX_DISP;
  const uint32_t* cc = reinterpret_cast<const uint32_t*>(costs);

  for (int p = 0; p < paths; ++p)
  {
    uint32_t c = cc[(p * step + offset) / 2 + threadIdx.x];
    cost0 += uint16_t((c >>  0) & 0x0000FFFF);
    cost1 += uint16_t((c >> 16) & 0x0000FFFF);
  }

  uint16_t cost = min(cost0, cost1);
  int d = threadIdx.x;

  // BlockMinIndex(cost, d, MAX_DISP / 2);
  // if (threadIdx.x == d) disparities[y * w + x] = (cost0 <= cost1) ? d0 : d1;

  uint16_t temp_cost = cost;
  int temp_d = d;

  BlockMinIndex(cost, d, MAX_DISP / 2);

  __syncthreads();

  if (threadIdx.x == d)
  {
    temp_cost = 10000;
  }

  BlockMinIndex(temp_cost, temp_d, MAX_DISP / 2);

  if (threadIdx.x == d)
  {
    disparities[y * w + x] = (uniqueness * temp_cost < cost &&
        abs(temp_d - d) > 1) ? 0 : (cost0 <= cost1) ? d0 : d1;
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void ComputeInvertedKernel(const uint16_t* __restrict__ costs,
    uint8_t* disparities, int paths, float uniqueness)
{
  const int w = gridDim.x;
  const int h = gridDim.y;
  const int x = blockIdx.x;
  const int y = blockIdx.y;
  int d = threadIdx.x;
  uint16_t cost = 0;

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

  uint16_t temp_cost = cost;
  int temp_d = d;

  BlockMinIndex(cost, d, MAX_DISP);

  __syncthreads();

  if (threadIdx.x == d)
  {
    temp_cost = 10000;
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
  const uint16_t* src = cost_->GetData();
  float u = uniqueness_;

  if (inverted_)
  {
    CUDA_LAUNCH(ComputeInvertedKernel<128>, grids, blocks, 0, 0, src, dst, p, u);
  }
  else
  {
    const dim3 blocks(d / 2);
    CUDA_LAUNCH(ComputeKernel<128>, grids, blocks, 0, 0, src, dst, p, u);
  }
}

} // namespace matchbox