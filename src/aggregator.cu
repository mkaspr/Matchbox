#include <matchbox/aggregator.h>
#include <matchbox/aggregate_cost.h>
#include <matchbox/device.h>
#include <matchbox/exception.h>
#include <matchbox/matching_cost.h>
#include <matchbox/util.cuh>

namespace matchbox
{

MATCHBOX_GLOBAL
void AggregateMatchingKernel(const uint8_t* __restrict__ matching_cost,
    uint16_t* aggregrate_cost, int count)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < count)
  {
    aggregrate_cost[index] = matching_cost[index];
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel(const uint8_t* __restrict__ matching_cost,
    uint16_t* aggregrate_cost, int w, int h, uint16_t P1, uint16_t P2)
{
  MATCHBOX_SHARED uint32_t shared[MAX_DISP + 2];

  const int y = blockIdx.x;
  const int k = threadIdx.x;

  int aggr = 0;
  int vmin = 0;

  shared[k] = 64 + P2;
  if (k >= MAX_DISP - 2) shared[k + 2] = 64 + P2;

  __syncthreads();

  for (int x = 0; x < w; ++x)
  {
    const int index = y * w * MAX_DISP + x * MAX_DISP + k;
    const uint8_t cost = matching_cost[index];

    const uint16_t left  = shared[k] + P1;
    const uint16_t right = shared[k + 2] + P1;
    const uint16_t far   = vmin + P2;

    aggr = cost + min(aggr, min(left, min(right, far))) - vmin;
    aggregrate_cost[index] = aggr;
    shared[k + 1] = aggr;

    vmin = (uint16_t)BlockMin((int)aggr, k, MAX_DISP);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel2(const uint8_t* __restrict__ matching_cost,
    uint16_t* aggregrate_cost, int w, int h, uint16_t P1, uint16_t P2)
{
  MATCHBOX_SHARED uint32_t shared[MAX_DISP + 2];

  const int y = blockIdx.x;
  const int k = threadIdx.x;

  shared[k] = 64 + P2;
  if (k >= MAX_DISP - 2) shared[k + 2] = 64 + P2;

  __syncthreads();

  int aggr = 0;
  int vmin = 0;

  for (int x = w - 1; x >= 0; --x)
  {
    const int index = y * w * MAX_DISP + x * MAX_DISP + k;
    const uint8_t cost = matching_cost[index];

    const uint16_t left  = shared[k] + P1;
    const uint16_t right = shared[k + 2] + P1;
    const uint16_t far   = vmin + P2;

    aggr = cost + min(aggr, min(left, min(right, far))) - vmin;
    aggregrate_cost[index] = aggr;
    shared[k + 1] = aggr;

    vmin = (uint16_t)BlockMin((int)aggr, k, MAX_DISP);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel3(const uint8_t* __restrict__ matching_cost,
    uint16_t* aggregrate_cost, int w, int h, uint16_t P1, uint16_t P2)
{
  MATCHBOX_SHARED uint32_t shared[MAX_DISP + 2];

  const int x = blockIdx.x;
  const int k = threadIdx.x;

  shared[k] = 64 + P2;
  if (k >= MAX_DISP - 2) shared[k + 2] = 64 + P2;

  __syncthreads();

  int aggr = 0;
  int vmin = 0;

  for (int y = 0; y < h; ++y)
  {
    const int index = y * w * MAX_DISP + x * MAX_DISP + k;
    const uint8_t cost = matching_cost[index];

    const uint16_t left  = shared[k] + P1;
    const uint16_t right = shared[k + 2] + P1;
    const uint16_t far   = vmin + P2;

    aggr = cost + min(aggr, min(left, min(right, far))) - vmin;
    aggregrate_cost[index] = aggr;
    shared[k + 1] = aggr;

    vmin = (uint16_t)BlockMin((int)aggr, k, MAX_DISP);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel4(const uint8_t* __restrict__ matching_cost,
    uint16_t* aggregrate_cost, int w, int h, uint16_t P1, uint16_t P2)
{
  MATCHBOX_SHARED uint32_t shared[MAX_DISP + 2];

  const int x = blockIdx.x;
  const int k = threadIdx.x;

  shared[k] = 64 + P2;
  if (k >= MAX_DISP - 2) shared[k + 2] = 64 + P2;

  __syncthreads();

  int aggr = 0;
  int vmin = 0;

  for (int y = h - 1; y >= 0; --y)
  {
    const int index = y * w * MAX_DISP + x * MAX_DISP + k;
    const uint8_t cost = matching_cost[index];

    const uint16_t left  = shared[k] + P1;
    const uint16_t right = shared[k + 2] + P1;
    const uint16_t far   = vmin + P2;

    aggr = cost + min(aggr, min(left, min(right, far))) - vmin;
    aggregrate_cost[index] = aggr;
    shared[k + 1] = aggr;

    vmin = (uint16_t)BlockMin((int)aggr, k, MAX_DISP);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel5(const uint8_t* __restrict__ matching_cost,
    uint16_t* aggregrate_cost, int w, int h, uint16_t P1, uint16_t P2)
{
  MATCHBOX_SHARED uint32_t shared[MAX_DISP + 2];

  const int k = threadIdx.x;
  const int ii = blockIdx.x;

  const int x = (ii < h) ? 0 : ii - h + 1;
  const int y = (ii < h) ? h - ii - 1 : 0;
  const int n = min(w - x, h - y);

  shared[k] = 64 + P2;
  if (k >= MAX_DISP - 2) shared[k + 2] = 64 + P2;

  __syncthreads();

  int aggr = 0;
  int vmin = 0;

  for (int i = 0; i < n; ++i)
  {
    const int xx = x + i;
    const int yy = y + i;
    const int index = yy * w * MAX_DISP + xx * MAX_DISP + k;
    const uint8_t cost = matching_cost[index];

    const uint16_t left  = shared[k] + P1;
    const uint16_t right = shared[k + 2] + P1;
    const uint16_t far   = vmin + P2;

    aggr = cost + min(aggr, min(left, min(right, far))) - vmin;
    aggregrate_cost[index] = aggr;
    shared[k + 1] = aggr;

    vmin = (uint16_t)BlockMin((int)aggr, k, MAX_DISP);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel6(const uint8_t* __restrict__ matching_cost,
    uint16_t* aggregrate_cost, int w, int h, uint16_t P1, uint16_t P2)
{
  MATCHBOX_SHARED uint32_t shared[MAX_DISP + 2];

  const int k = threadIdx.x;
  const int ii = blockIdx.x;

  const int x = (ii < h) ? 0 : ii - h + 1;
  const int y = (ii < h) ? h - ii - 1 : 0;
  const int n = min(w - x, h - y);

  shared[k] = 64 + P2;
  if (k >= MAX_DISP - 2) shared[k + 2] = 64 + P2;

  __syncthreads();

  int aggr = 0;
  int vmin = 0;

  for (int i = 0; i < n; ++i)
  {
    const int xx = w - x - i - 1;
    const int yy = h - y - i - 1;
    const int index = yy * w * MAX_DISP + xx * MAX_DISP + k;
    const uint8_t cost = matching_cost[index];

    const uint16_t left  = shared[k] + P1;
    const uint16_t right = shared[k + 2] + P1;
    const uint16_t far   = vmin + P2;

    aggr = cost + min(aggr, min(left, min(right, far))) - vmin;
    aggregrate_cost[index] = aggr;
    shared[k + 1] = aggr;

    vmin = (uint16_t)BlockMin((int)aggr, k, MAX_DISP);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel7(const uint8_t* __restrict__ matching_cost,
    uint16_t* aggregrate_cost, int w, int h, uint16_t P1, uint16_t P2)
{
  MATCHBOX_SHARED uint32_t shared[MAX_DISP + 2];

  const int k = threadIdx.x;
  const int ii = blockIdx.x;

  const int x = (ii < h) ? 0 : ii - h + 1;
  const int y = (ii < h) ? h - ii - 1 : 0;
  const int n = min(w - x, h - y);

  shared[k] = 64 + P2;
  if (k >= MAX_DISP - 2) shared[k + 2] = 64 + P2;

  __syncthreads();

  int aggr = 0;
  int vmin = 0;

  for (int i = 0; i < n; ++i)
  {
    const int xx = x + i;
    const int yy = h - y - i - 1;
    const int index = yy * w * MAX_DISP + xx * MAX_DISP + k;
    const uint8_t cost = matching_cost[index];

    const uint16_t left  = shared[k] + P1;
    const uint16_t right = shared[k + 2] + P1;
    const uint16_t far   = vmin + P2;

    aggr = cost + min(aggr, min(left, min(right, far))) - vmin;
    aggregrate_cost[index] = aggr;
    shared[k + 1] = aggr;

    vmin = (uint16_t)BlockMin((int)aggr, k, MAX_DISP);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel8(const uint8_t* __restrict__ matching_cost,
    uint16_t* aggregrate_cost, int w, int h, uint16_t P1, uint16_t P2)
{
  MATCHBOX_SHARED uint32_t shared[MAX_DISP + 2];

  const int k = threadIdx.x;
  const int ii = blockIdx.x;

  const int x = (ii < h) ? 0 : ii - h + 1;
  const int y = (ii < h) ? h - ii - 1 : 0;
  const int n = min(w - x, h - y);

  shared[k] = 64 + P2;
  if (k >= MAX_DISP - 2) shared[k + 2] = 64 + P2;

  __syncthreads();

  int aggr = 0;
  int vmin = 0;

  for (int i = 0; i < n; ++i)
  {
    const int xx = w - x - i - 1;
    const int yy = y + i;
    const int index = yy * w * MAX_DISP + xx * MAX_DISP + k;
    const uint8_t cost = matching_cost[index];

    const uint16_t left  = shared[k] + P1;
    const uint16_t right = shared[k + 2] + P1;
    const uint16_t far   = vmin + P2;

    aggr = cost + min(aggr, min(left, min(right, far))) - vmin;
    aggregrate_cost[index] = aggr;
    shared[k + 1] = aggr;

    vmin = (uint16_t)BlockMin((int)aggr, k, MAX_DISP);
  }
}

Aggregator::Aggregator(std::shared_ptr<const MatchingCost> matching_cost) :
  matching_cost_(matching_cost),
  degree_(3)
{
}

std::shared_ptr<const MatchingCost> Aggregator::GetMatchingCost() const
{
  return matching_cost_;
}

int Aggregator::GetDegree() const
{
  return degree_;
}

void Aggregator::SetDegree(int degree)
{
  MATCHBOX_DEBUG(degree >= 0 && degree <= 3);
  degree_ = degree;
}

void Aggregator::Aggregate(AggregateCost& cost) const
{
  ResizeCost(cost);

  switch (degree_)
  {
    case 3: AggregateDiagonal(cost);
    case 2: AggregateVertical(cost);
    case 1: AggregateHorizontal(cost);

      break;

    case 0: AggregateMatching(cost);
  }
}

int Aggregator::GetPathCount() const
{
  switch (degree_)
  {
    case 0: return 1;
    case 1: return 2;
    case 2: return 4;
    case 3: return 8;
  }

  MATCHBOX_THROW("invalid degree");
  DEVICE_RETURN(-1);
}

void Aggregator::ResizeCost(AggregateCost& cost) const
{
  const int w = matching_cost_->GetWidth();
  const int h = matching_cost_->GetHeight();
  const int d = matching_cost_->GetDepth();
  const int p = GetPathCount();
  cost.SetSize(w, h, d, p);
}

void Aggregator::AggregateMatching(AggregateCost& cost) const
{
  const int w = matching_cost_->GetWidth();
  const int h = matching_cost_->GetHeight();
  const int d = matching_cost_->GetDepth();

  const int blocks = 512;
  const int threads = w * h * d;
  const int grids = GetGrids(threads, blocks);
  const uint8_t* src = matching_cost_->GetData();
  uint16_t* dst = cost.GetData();

  CUDA_LAUNCH(AggregateMatchingKernel, grids, blocks, 0, 0, src, dst, threads);
}

void Aggregator::AggregateHorizontal(AggregateCost& cost) const
{
  MATCHBOX_DEBUG(matching_cost_->GetDepth() == 128);

  const int w = matching_cost_->GetWidth();
  const int h = matching_cost_->GetHeight();
  const int d = matching_cost_->GetDepth();

  const int grids = h;
  const int blocks = d;
  const uint8_t* src = matching_cost_->GetData();
  uint16_t* dst = cost.GetData();

  const int offset = w * h * d;

  CUDA_LAUNCH(AggregateKernel<128>, grids, blocks, 0, 0, src, dst,
      w, h, 20, 100);

  CUDA_LAUNCH(AggregateKernel2<128>, grids, blocks, 0, 0, src,
      dst + offset, w, h, 20, 100);
}

void Aggregator::AggregateVertical(AggregateCost& cost) const
{
  MATCHBOX_DEBUG(matching_cost_->GetDepth() == 128);

  const int w = matching_cost_->GetWidth();
  const int h = matching_cost_->GetHeight();
  const int d = matching_cost_->GetDepth();

  const int grids = w;
  const int blocks = d;
  const uint8_t* src = matching_cost_->GetData();
  uint16_t* dst = cost.GetData();

  int offset = 2 * w * h * d;

  CUDA_LAUNCH(AggregateKernel3<128>, grids, blocks, 0, 0, src,
      dst + offset, w, h, 20, 100);

  offset = 3 * w * h * d;

  CUDA_LAUNCH(AggregateKernel4<128>, grids, blocks, 0, 0, src,
      dst + offset, w, h, 20, 100);
}

void Aggregator::AggregateDiagonal(AggregateCost& cost) const
{
  MATCHBOX_DEBUG(matching_cost_->GetDepth() == 128);

  const int w = matching_cost_->GetWidth();
  const int h = matching_cost_->GetHeight();
  const int d = matching_cost_->GetDepth();

  const int blocks = d;
  const int grids = w + h - 1;
  const uint8_t* src = matching_cost_->GetData();
  uint16_t* dst = cost.GetData();

  int offset = 4 * w * h * d;

  CUDA_LAUNCH(AggregateKernel5<128>, grids, blocks, 0, 0, src,
      dst + offset, w, h, 20, 100);

  offset = 5 * w * h * d;

  CUDA_LAUNCH(AggregateKernel6<128>, grids, blocks, 0, 0, src,
      dst + offset, w, h, 20, 100);

  offset = 6 * w * h * d;

  CUDA_LAUNCH(AggregateKernel7<128>, grids, blocks, 0, 0, src,
      dst + offset, w, h, 20, 100);

  offset = 7 * w * h * d;

  CUDA_LAUNCH(AggregateKernel8<128>, grids, blocks, 0, 0, src,
      dst + offset, w, h, 20, 100);
}

} // namespace matchbox