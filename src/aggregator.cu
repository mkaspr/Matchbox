#include <matchbox/aggregator.h>
#include <bitset>
#include <matchbox/aggregate_cost.h>
#include <matchbox/exception.h>
#include <matchbox/matching_cost.h>
#include <matchbox/util.cuh>

namespace matchbox
{

MATCHBOX_DEVICE
inline uint32_t WarpMin2(uint32_t value)
{
  value = ((value & 0xFFFF) > (value >> 16)) ?
        (value & 0xFFFF0000) | (value >> 16) :
        (value & 0x0000FFFF) | (value << 16);

  for (int i = 16; i > 0; i >>= 1)
  {
    value = __vminu2(value, __shfl_down(value, i, 32));
  }

  return __shfl(value, 0, 32);
}

MATCHBOX_GLOBAL
void AggregateMatchingKernel(const uint8_t* __restrict__ matching_cost,
    uint16_t* __restrict__ aggregate_cost, int count)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < count)
  {
    aggregate_cost[index] = matching_cost[index];
  }
}

// template <int MAX_DISP>
// MATCHBOX_GLOBAL
// void AggregateKernel(const uint8_t* __restrict__ matching_cost,
//     uint16_t* __restrict__ aggregate_cost, int w, int h, uint8_t P1, uint8_t P2)
// {
//   uint32_t shared[(MAX_DISP >> 1) + 2];
//
//   const int k = 4 * threadIdx.x;
//   const int y = 2 * blockIdx.x + threadIdx.y;
//   const uint32_t* mc = reinterpret_cast<const uint32_t*>(matching_cost);
//   uint32_t* ac = reinterpret_cast<uint32_t*>(aggregate_cost);
//
//   uint32_t half_aggr_l = 0;
//   uint32_t half_aggr_h = 0;
//   uint32_t half_min = 0;
//
//   const uint32_t max_cost = 64 + P2;
//   const uint32_t v2_max_cost = __byte_perm(max_cost, 0, 0x4040);
//   const uint32_t v2_P1 = __byte_perm(P1, 0, 0x4040);
//   const uint32_t v2_P2 = __byte_perm(P2, 0, 0x4040);
//
//   shared[2 * threadIdx.x + 0] = v2_max_cost;
//   shared[2 * threadIdx.x + 1] = v2_max_cost;
//   shared[2 * threadIdx.x + 2] = v2_max_cost;
//
//   for (int x = 0; x < w; ++x)
//   {
//     const int index = y * w * MAX_DISP + x * MAX_DISP + k;
//     const uint32_t costs = mc[index >> 2];
//
//     const uint32_t half_costs_l = __byte_perm(costs, 0, 0x4140);
//     const uint32_t half_costs_h = __byte_perm(costs, 0, 0x4342);
//
//     const uint32_t half_left  = __vadd2(shared[2 * threadIdx.x + 0], v2_P1);
//     const uint32_t half_mid   = __vadd2(shared[2 * threadIdx.x + 1], v2_P1);
//     const uint32_t half_right = __vadd2(shared[2 * threadIdx.x + 2], v2_P1);
//     const uint32_t half_far   = __vadd2(half_min, v2_P2);
//
//     half_aggr_l = __vadd2(half_costs_l, __vsub2(__vminu2(half_aggr_l, __vminu2(half_left, __vminu2(half_mid, half_far))), half_min));
//     half_aggr_h = __vadd2(half_costs_h, __vsub2(__vminu2(half_aggr_h, __vminu2(half_mid, __vminu2(half_right, half_far))), half_min));
//
//     ac[(index >> 1) + 0] = half_aggr_l;
//     ac[(index >> 1) + 1] = half_aggr_h;
//
//     shared[2 * threadIdx.x + 1] = half_aggr_l;
//     shared[2 * threadIdx.x + 2] = half_aggr_h;
//
//     half_min = WarpMin2(__vminu2(half_aggr_l, half_aggr_h));
//   }
// }

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel1(const uint8_t* __restrict__ matching_cost,
    uint16_t* __restrict__ aggregate_cost, int w, int h, uint8_t P1, uint8_t P2)
{
  MATCHBOX_SHARED uint32_t shared[2 * (MAX_DISP + 2)];

  const int k = 4 * threadIdx.x;
  const int y = 2 * blockIdx.x + threadIdx.y;
  const int o = threadIdx.y * (MAX_DISP + 2);
  const uint32_t* mc = reinterpret_cast<const uint32_t*>(matching_cost);

  int aggr[] = { 0, 0, 0, 0 };
  int vmin = 0;

  const uint32_t max_cost = 64 + P2;
  shared[o + k + 0] = max_cost;
  shared[o + k + 1] = max_cost;
  shared[o + k + 2] = max_cost;
  shared[o + k + 3] = max_cost;
  shared[o + k + 5] = max_cost;
  shared[o + k + 6] = max_cost;

  __syncthreads();

  for (int x = 0; x < w; ++x)
  {
    const int index = y * w * MAX_DISP + x * MAX_DISP + k;
    const uint32_t costs = mc[index >> 2];
    const uint8_t cost0 = (costs >>  0) & 0xFF;
    const uint8_t cost1 = (costs >>  8) & 0xFF;
    const uint8_t cost2 = (costs >> 16) & 0xFF;
    const uint8_t cost3 = (costs >> 24) & 0xFF;

    const uint8_t l0 = shared[o + k + 0] + P1;
    const uint8_t l1 = shared[o + k + 1] + P1;
    const uint8_t l2 = shared[o + k + 2] + P1;
    const uint8_t l3 = shared[o + k + 3] + P1;

    const uint8_t r0 = shared[o + k + 2] + P1;
    const uint8_t r1 = shared[o + k + 3] + P1;
    const uint8_t r2 = shared[o + k + 4] + P1;
    const uint8_t r3 = shared[o + k + 5] + P1;

    const uint8_t f = vmin + P2;

    aggr[0] = cost0 + min(aggr[0], min(l0, min(r0, f))) - vmin;
    aggr[1] = cost1 + min(aggr[1], min(l1, min(r1, f))) - vmin;
    aggr[2] = cost2 + min(aggr[2], min(l2, min(r2, f))) - vmin;
    aggr[3] = cost3 + min(aggr[3], min(l3, min(r3, f))) - vmin;

    aggregate_cost[index + 0] = aggr[0];
    aggregate_cost[index + 1] = aggr[1];
    aggregate_cost[index + 2] = aggr[2];
    aggregate_cost[index + 3] = aggr[3];

    __syncthreads();

    shared[o + k + 1] = aggr[0];
    shared[o + k + 2] = aggr[1];
    shared[o + k + 3] = aggr[2];
    shared[o + k + 4] = aggr[3];

    const int a = min(aggr[0], min(aggr[1], min(aggr[2], aggr[3])));
    vmin = WarpMin(a);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel2(const uint8_t* __restrict__ matching_cost,
    uint16_t* __restrict__ aggregate_cost, int w, int h, uint8_t P1, uint8_t P2)
{
  MATCHBOX_SHARED uint32_t shared[2 * (MAX_DISP + 2)];

  const int k = 4 * threadIdx.x;
  const int y = 2 * blockIdx.x + threadIdx.y;
  const int o = threadIdx.y * (MAX_DISP + 2);
  const uint32_t* mc = reinterpret_cast<const uint32_t*>(matching_cost);

  int aggr[] = { 0, 0, 0, 0 };
  int vmin = 0;

  const uint32_t max_cost = 64 + P2;
  shared[o + k + 0] = max_cost;
  shared[o + k + 1] = max_cost;
  shared[o + k + 2] = max_cost;
  shared[o + k + 3] = max_cost;
  shared[o + k + 5] = max_cost;
  shared[o + k + 6] = max_cost;

  __syncthreads();

  for (int x = w - 1; x >= 0; --x)
  {
    const int index = y * w * MAX_DISP + x * MAX_DISP + k;
    const uint32_t costs = mc[index >> 2];
    const uint8_t cost0 = (costs >>  0) & 0xFF;
    const uint8_t cost1 = (costs >>  8) & 0xFF;
    const uint8_t cost2 = (costs >> 16) & 0xFF;
    const uint8_t cost3 = (costs >> 24) & 0xFF;

    const uint8_t l0 = shared[o + k + 0] + P1;
    const uint8_t l1 = shared[o + k + 1] + P1;
    const uint8_t l2 = shared[o + k + 2] + P1;
    const uint8_t l3 = shared[o + k + 3] + P1;

    const uint8_t r0 = shared[o + k + 2] + P1;
    const uint8_t r1 = shared[o + k + 3] + P1;
    const uint8_t r2 = shared[o + k + 4] + P1;
    const uint8_t r3 = shared[o + k + 5] + P1;

    const uint8_t f = vmin + P2;

    aggr[0] = cost0 + min(aggr[0], min(l0, min(r0, f))) - vmin;
    aggr[1] = cost1 + min(aggr[1], min(l1, min(r1, f))) - vmin;
    aggr[2] = cost2 + min(aggr[2], min(l2, min(r2, f))) - vmin;
    aggr[3] = cost3 + min(aggr[3], min(l3, min(r3, f))) - vmin;

    aggregate_cost[index + 0] += aggr[0];
    aggregate_cost[index + 1] += aggr[1];
    aggregate_cost[index + 2] += aggr[2];
    aggregate_cost[index + 3] += aggr[3];

    __syncthreads();

    shared[o + k + 1] = aggr[0];
    shared[o + k + 2] = aggr[1];
    shared[o + k + 3] = aggr[2];
    shared[o + k + 4] = aggr[3];

    const int a = min(aggr[0], min(aggr[1], min(aggr[2], aggr[3])));
    vmin = WarpMin(a);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel3(const uint8_t* __restrict__ matching_cost,
    uint16_t* __restrict__ aggregate_cost, int w, int h, uint8_t P1, uint8_t P2)
{
  MATCHBOX_SHARED uint32_t shared[2 * (MAX_DISP + 2)];

  const int k = 4 * threadIdx.x;
  const int x = 2 * blockIdx.x + threadIdx.y;
  const int o = threadIdx.y * (MAX_DISP + 2);
  const uint32_t* mc = reinterpret_cast<const uint32_t*>(matching_cost);

  int aggr[] = { 0, 0, 0, 0 };
  int vmin = 0;

  const uint32_t max_cost = 64 + P2;
  shared[o + k + 0] = max_cost;
  shared[o + k + 1] = max_cost;
  shared[o + k + 2] = max_cost;
  shared[o + k + 3] = max_cost;
  shared[o + k + 5] = max_cost;
  shared[o + k + 6] = max_cost;

  __syncthreads();

  for (int y = 0; y < h; ++y)
  {
    const int index = y * w * MAX_DISP + x * MAX_DISP + k;
    const uint32_t costs = mc[index >> 2];
    const uint8_t cost0 = (costs >>  0) & 0xFF;
    const uint8_t cost1 = (costs >>  8) & 0xFF;
    const uint8_t cost2 = (costs >> 16) & 0xFF;
    const uint8_t cost3 = (costs >> 24) & 0xFF;

    const uint8_t l0 = shared[o + k + 0] + P1;
    const uint8_t l1 = shared[o + k + 1] + P1;
    const uint8_t l2 = shared[o + k + 2] + P1;
    const uint8_t l3 = shared[o + k + 3] + P1;

    const uint8_t r0 = shared[o + k + 2] + P1;
    const uint8_t r1 = shared[o + k + 3] + P1;
    const uint8_t r2 = shared[o + k + 4] + P1;
    const uint8_t r3 = shared[o + k + 5] + P1;

    const uint8_t f = vmin + P2;

    aggr[0] = cost0 + min(aggr[0], min(l0, min(r0, f))) - vmin;
    aggr[1] = cost1 + min(aggr[1], min(l1, min(r1, f))) - vmin;
    aggr[2] = cost2 + min(aggr[2], min(l2, min(r2, f))) - vmin;
    aggr[3] = cost3 + min(aggr[3], min(l3, min(r3, f))) - vmin;

    aggregate_cost[index + 0] += aggr[0];
    aggregate_cost[index + 1] += aggr[1];
    aggregate_cost[index + 2] += aggr[2];
    aggregate_cost[index + 3] += aggr[3];

    __syncthreads();

    shared[o + k + 1] = aggr[0];
    shared[o + k + 2] = aggr[1];
    shared[o + k + 3] = aggr[2];
    shared[o + k + 4] = aggr[3];

    const int a = min(aggr[0], min(aggr[1], min(aggr[2], aggr[3])));
    vmin = WarpMin(a);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel4(const uint8_t* __restrict__ matching_cost,
    uint16_t* __restrict__ aggregate_cost, int w, int h, uint8_t P1, uint8_t P2)
{
  MATCHBOX_SHARED uint32_t shared[2 * (MAX_DISP + 2)];

  const int k = 4 * threadIdx.x;
  const int x = 2 * blockIdx.x + threadIdx.y;
  const int o = threadIdx.y * (MAX_DISP + 2);
  const uint32_t* mc = reinterpret_cast<const uint32_t*>(matching_cost);

  int aggr[] = { 0, 0, 0, 0 };
  int vmin = 0;

  const uint32_t max_cost = 64 + P2;
  shared[o + k + 0] = max_cost;
  shared[o + k + 1] = max_cost;
  shared[o + k + 2] = max_cost;
  shared[o + k + 3] = max_cost;
  shared[o + k + 5] = max_cost;
  shared[o + k + 6] = max_cost;

  __syncthreads();

  for (int y = h - 1; y >= 0; --y)
  {
    const int index = y * w * MAX_DISP + x * MAX_DISP + k;
    const uint32_t costs = mc[index >> 2];
    const uint8_t cost0 = (costs >>  0) & 0xFF;
    const uint8_t cost1 = (costs >>  8) & 0xFF;
    const uint8_t cost2 = (costs >> 16) & 0xFF;
    const uint8_t cost3 = (costs >> 24) & 0xFF;

    const uint8_t l0 = shared[o + k + 0] + P1;
    const uint8_t l1 = shared[o + k + 1] + P1;
    const uint8_t l2 = shared[o + k + 2] + P1;
    const uint8_t l3 = shared[o + k + 3] + P1;

    const uint8_t r0 = shared[o + k + 2] + P1;
    const uint8_t r1 = shared[o + k + 3] + P1;
    const uint8_t r2 = shared[o + k + 4] + P1;
    const uint8_t r3 = shared[o + k + 5] + P1;
    const uint8_t f = vmin + P2;

    aggr[0] = cost0 + min(aggr[0], min(l0, min(r0, f))) - vmin;
    aggr[1] = cost1 + min(aggr[1], min(l1, min(r1, f))) - vmin;
    aggr[2] = cost2 + min(aggr[2], min(l2, min(r2, f))) - vmin;
    aggr[3] = cost3 + min(aggr[3], min(l3, min(r3, f))) - vmin;

    aggregate_cost[index + 0] += aggr[0];
    aggregate_cost[index + 1] += aggr[1];
    aggregate_cost[index + 2] += aggr[2];
    aggregate_cost[index + 3] += aggr[3];

    __syncthreads();

    shared[o + k + 1] = aggr[0];
    shared[o + k + 2] = aggr[1];
    shared[o + k + 3] = aggr[2];
    shared[o + k + 4] = aggr[3];

    const int a = min(aggr[0], min(aggr[1], min(aggr[2], aggr[3])));
    vmin = WarpMin(a);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel5(const uint8_t* __restrict__ matching_cost,
    uint16_t* __restrict__ aggregate_cost, int w, int h, uint8_t P1, uint8_t P2)
{
  MATCHBOX_SHARED uint32_t shared[2 * (MAX_DISP + 2)];

  const int k = 4 * threadIdx.x;
  const int ii = 2 * blockIdx.x + threadIdx.y;
  const int o = threadIdx.y * (MAX_DISP + 2);
  if (ii >= h + w) return;

  const uint32_t* mc = reinterpret_cast<const uint32_t*>(matching_cost);

  const int x = (ii < h) ? 0 : ii - h + 1;
  const int y = (ii < h) ? h - ii - 1 : 0;
  const int n = min(w - x, h - y);

  int aggr[] = { 0, 0, 0, 0 };
  int vmin = 0;

  const uint32_t max_cost = 64 + P2;
  shared[o + k + 0] = max_cost;
  shared[o + k + 1] = max_cost;
  shared[o + k + 2] = max_cost;
  shared[o + k + 3] = max_cost;
  shared[o + k + 5] = max_cost;
  shared[o + k + 6] = max_cost;

  __syncthreads();

  for (int i = 0; i < n; ++i)
  {
    const int xx = x + i;
    const int yy = y + i;
    const int index = yy * w * MAX_DISP + xx * MAX_DISP + k;
    const uint32_t costs = mc[index >> 2];
    const uint8_t cost0 = (costs >>  0) & 0xFF;
    const uint8_t cost1 = (costs >>  8) & 0xFF;
    const uint8_t cost2 = (costs >> 16) & 0xFF;
    const uint8_t cost3 = (costs >> 24) & 0xFF;

    const uint8_t l0 = shared[o + k + 0] + P1;
    const uint8_t l1 = shared[o + k + 1] + P1;
    const uint8_t l2 = shared[o + k + 2] + P1;
    const uint8_t l3 = shared[o + k + 3] + P1;

    const uint8_t r0 = shared[o + k + 2] + P1;
    const uint8_t r1 = shared[o + k + 3] + P1;
    const uint8_t r2 = shared[o + k + 4] + P1;
    const uint8_t r3 = shared[o + k + 5] + P1;

    const uint8_t f = vmin + P2;

    aggr[0] = cost0 + min(aggr[0], min(l0, min(r0, f))) - vmin;
    aggr[1] = cost1 + min(aggr[1], min(l1, min(r1, f))) - vmin;
    aggr[2] = cost2 + min(aggr[2], min(l2, min(r2, f))) - vmin;
    aggr[3] = cost3 + min(aggr[3], min(l3, min(r3, f))) - vmin;

    aggregate_cost[index + 0] += aggr[0];
    aggregate_cost[index + 1] += aggr[1];
    aggregate_cost[index + 2] += aggr[2];
    aggregate_cost[index + 3] += aggr[3];

    __syncthreads();

    shared[o + k + 1] = aggr[0];
    shared[o + k + 2] = aggr[1];
    shared[o + k + 3] = aggr[2];
    shared[o + k + 4] = aggr[3];

    const int a = min(aggr[0], min(aggr[1], min(aggr[2], aggr[3])));
    vmin = WarpMin(a);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel6(const uint8_t* __restrict__ matching_cost,
    uint16_t* __restrict__ aggregate_cost, int w, int h, uint8_t P1, uint8_t P2)
{
  MATCHBOX_SHARED uint32_t shared[2 * (MAX_DISP + 2)];

  const int k = 4 * threadIdx.x;
  const int ii = 2 * blockIdx.x + threadIdx.y;
  const int o = threadIdx.y * (MAX_DISP + 2);
  if (ii >= h + w) return;

  const uint32_t* mc = reinterpret_cast<const uint32_t*>(matching_cost);

  const int x = (ii < h) ? 0 : ii - h + 1;
  const int y = (ii < h) ? h - ii - 1 : 0;
  const int n = min(w - x, h - y);

  int aggr[] = { 0, 0, 0, 0 };
  int vmin = 0;

  const uint32_t max_cost = 64 + P2;
  shared[o + k + 0] = max_cost;
  shared[o + k + 1] = max_cost;
  shared[o + k + 2] = max_cost;
  shared[o + k + 3] = max_cost;
  shared[o + k + 5] = max_cost;
  shared[o + k + 6] = max_cost;

  __syncthreads();

  for (int i = 0; i < n; ++i)
  {
    const int xx = w - x - i - 1;
    const int yy = h - y - i - 1;
    const int index = yy * w * MAX_DISP + xx * MAX_DISP + k;
    const uint32_t costs = mc[index >> 2];
    const uint8_t cost0 = (costs >>  0) & 0xFF;
    const uint8_t cost1 = (costs >>  8) & 0xFF;
    const uint8_t cost2 = (costs >> 16) & 0xFF;
    const uint8_t cost3 = (costs >> 24) & 0xFF;

    const uint8_t l0 = shared[o + k + 0] + P1;
    const uint8_t l1 = shared[o + k + 1] + P1;
    const uint8_t l2 = shared[o + k + 2] + P1;
    const uint8_t l3 = shared[o + k + 3] + P1;

    const uint8_t r0 = shared[o + k + 2] + P1;
    const uint8_t r1 = shared[o + k + 3] + P1;
    const uint8_t r2 = shared[o + k + 4] + P1;
    const uint8_t r3 = shared[o + k + 5] + P1;

    const uint8_t f = vmin + P2;

    aggr[0] = cost0 + min(aggr[0], min(l0, min(r0, f))) - vmin;
    aggr[1] = cost1 + min(aggr[1], min(l1, min(r1, f))) - vmin;
    aggr[2] = cost2 + min(aggr[2], min(l2, min(r2, f))) - vmin;
    aggr[3] = cost3 + min(aggr[3], min(l3, min(r3, f))) - vmin;

    aggregate_cost[index + 0] += aggr[0];
    aggregate_cost[index + 1] += aggr[1];
    aggregate_cost[index + 2] += aggr[2];
    aggregate_cost[index + 3] += aggr[3];

    __syncthreads();

    shared[o + k + 1] = aggr[0];
    shared[o + k + 2] = aggr[1];
    shared[o + k + 3] = aggr[2];
    shared[o + k + 4] = aggr[3];

    const int a = min(aggr[0], min(aggr[1], min(aggr[2], aggr[3])));
    vmin = WarpMin(a);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel7(const uint8_t* __restrict__ matching_cost,
    uint16_t* __restrict__ aggregate_cost, int w, int h, uint8_t P1, uint8_t P2)
{
  MATCHBOX_SHARED uint32_t shared[2 * (MAX_DISP + 2)];

  const int k = 4 * threadIdx.x;
  const int ii = 2 * blockIdx.x + threadIdx.y;
  const int o = threadIdx.y * (MAX_DISP + 2);
  if (ii >= h + w) return;

  const uint32_t* mc = reinterpret_cast<const uint32_t*>(matching_cost);

  const int x = (ii < h) ? 0 : ii - h + 1;
  const int y = (ii < h) ? h - ii - 1 : 0;
  const int n = min(w - x, h - y);

  int aggr[] = { 0, 0, 0, 0 };
  int vmin = 0;

  const uint32_t max_cost = 64 + P2;
  shared[o + k + 0] = max_cost;
  shared[o + k + 1] = max_cost;
  shared[o + k + 2] = max_cost;
  shared[o + k + 3] = max_cost;
  shared[o + k + 5] = max_cost;
  shared[o + k + 6] = max_cost;

  __syncthreads();

  for (int i = 0; i < n; ++i)
  {
    const int xx = x + i;
    const int yy = h - y - i - 1;
    const int index = yy * w * MAX_DISP + xx * MAX_DISP + k;
    const uint32_t costs = mc[index >> 2];
    const uint8_t cost0 = (costs >>  0) & 0xFF;
    const uint8_t cost1 = (costs >>  8) & 0xFF;
    const uint8_t cost2 = (costs >> 16) & 0xFF;
    const uint8_t cost3 = (costs >> 24) & 0xFF;

    const uint8_t l0 = shared[o + k + 0] + P1;
    const uint8_t l1 = shared[o + k + 1] + P1;
    const uint8_t l2 = shared[o + k + 2] + P1;
    const uint8_t l3 = shared[o + k + 3] + P1;

    const uint8_t r0 = shared[o + k + 2] + P1;
    const uint8_t r1 = shared[o + k + 3] + P1;
    const uint8_t r2 = shared[o + k + 4] + P1;
    const uint8_t r3 = shared[o + k + 5] + P1;

    const uint8_t f = vmin + P2;

    aggr[0] = cost0 + min(aggr[0], min(l0, min(r0, f))) - vmin;
    aggr[1] = cost1 + min(aggr[1], min(l1, min(r1, f))) - vmin;
    aggr[2] = cost2 + min(aggr[2], min(l2, min(r2, f))) - vmin;
    aggr[3] = cost3 + min(aggr[3], min(l3, min(r3, f))) - vmin;

    aggregate_cost[index + 0] += aggr[0];
    aggregate_cost[index + 1] += aggr[1];
    aggregate_cost[index + 2] += aggr[2];
    aggregate_cost[index + 3] += aggr[3];

    __syncthreads();

    shared[o + k + 1] = aggr[0];
    shared[o + k + 2] = aggr[1];
    shared[o + k + 3] = aggr[2];
    shared[o + k + 4] = aggr[3];

    const int a = min(aggr[0], min(aggr[1], min(aggr[2], aggr[3])));
    vmin = WarpMin(a);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel8(const uint8_t* __restrict__ matching_cost,
    uint16_t* __restrict__ aggregate_cost, int w, int h, uint8_t P1, uint8_t P2)
{
  MATCHBOX_SHARED uint32_t shared[2 * (MAX_DISP + 2)];

  const int k = 4 * threadIdx.x;
  const int ii = 2 * blockIdx.x + threadIdx.y;
  const int o = threadIdx.y * (MAX_DISP + 2);
  if (ii >= h + w) return;

  const uint32_t* mc = reinterpret_cast<const uint32_t*>(matching_cost);

  const int x = (ii < h) ? 0 : ii - h + 1;
  const int y = (ii < h) ? h - ii - 1 : 0;
  const int n = min(w - x, h - y);

  int aggr[] = { 0, 0, 0, 0 };
  int vmin = 0;

  const uint32_t max_cost = 64 + P2;
  shared[o + k + 0] = max_cost;
  shared[o + k + 1] = max_cost;
  shared[o + k + 2] = max_cost;
  shared[o + k + 3] = max_cost;
  shared[o + k + 5] = max_cost;
  shared[o + k + 6] = max_cost;

  __syncthreads();

  for (int i = 0; i < n; ++i)
  {
    const int xx = w - x - i - 1;
    const int yy = y + i;
    const int index = yy * w * MAX_DISP + xx * MAX_DISP + k;
    const uint32_t costs = mc[index >> 2];
    const uint8_t cost0 = (costs >>  0) & 0xFF;
    const uint8_t cost1 = (costs >>  8) & 0xFF;
    const uint8_t cost2 = (costs >> 16) & 0xFF;
    const uint8_t cost3 = (costs >> 24) & 0xFF;

    const uint8_t l0 = shared[o + k + 0] + P1;
    const uint8_t l1 = shared[o + k + 1] + P1;
    const uint8_t l2 = shared[o + k + 2] + P1;
    const uint8_t l3 = shared[o + k + 3] + P1;

    const uint8_t r0 = shared[o + k + 2] + P1;
    const uint8_t r1 = shared[o + k + 3] + P1;
    const uint8_t r2 = shared[o + k + 4] + P1;
    const uint8_t r3 = shared[o + k + 5] + P1;

    const uint8_t f = vmin + P2;

    aggr[0] = cost0 + min(aggr[0], min(l0, min(r0, f))) - vmin;
    aggr[1] = cost1 + min(aggr[1], min(l1, min(r1, f))) - vmin;
    aggr[2] = cost2 + min(aggr[2], min(l2, min(r2, f))) - vmin;
    aggr[3] = cost3 + min(aggr[3], min(l3, min(r3, f))) - vmin;

    aggregate_cost[index + 0] += aggr[0];
    aggregate_cost[index + 1] += aggr[1];
    aggregate_cost[index + 2] += aggr[2];
    aggregate_cost[index + 3] += aggr[3];

    __syncthreads();

    shared[o + k + 1] = aggr[0];
    shared[o + k + 2] = aggr[1];
    shared[o + k + 3] = aggr[2];
    shared[o + k + 4] = aggr[3];

    const int a = min(aggr[0], min(aggr[1], min(aggr[2], aggr[3])));
    vmin = WarpMin(a);
  }
}

Aggregator::Aggregator(std::shared_ptr<const MatchingCost> matching_cost) :
  matching_cost_(matching_cost),
  directions_(DIR_ALL)
{
  Initialize();
}

Aggregator::~Aggregator()
{
  for (int i = 0; i < stream_count; ++i)
  {
    cudaStreamDestroy(streams[i]);
  }
}

std::shared_ptr<const MatchingCost> Aggregator::GetMatchingCost() const
{
  return matching_cost_;
}

int Aggregator::GetDirections() const
{
  return directions_;
}

void Aggregator::SetDirections(int directions)
{
  directions_ = directions & DIR_ALL;
}

void Aggregator::Aggregate(AggregateCost& cost) const
{
  ResizeCost(cost);
  AggregateMatching(cost);
  AggregateHorizontal(cost);
  AggregateVertical(cost);
  AggregateDiagonal(cost);
}

int Aggregator::GetPathCount() const
{
  return std::bitset<32>(directions_).count();
}

void Aggregator::ResizeCost(AggregateCost& cost) const
{
  const int w = matching_cost_->GetWidth();
  const int h = matching_cost_->GetHeight();
  const int d = matching_cost_->GetDepth();
  cost.SetSize(w, h, d);
  cost.Clear(); // <========================== TODO: remove
}

void Aggregator::AggregateMatching(AggregateCost& cost) const
{
  if (directions_ == DIR_NONE)
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
}

void Aggregator::AggregateHorizontal(AggregateCost& cost) const
{
  MATCHBOX_DEBUG(matching_cost_->GetDepth() == 128);

  const int w = matching_cost_->GetWidth();
  const int h = matching_cost_->GetHeight();
  const int d = matching_cost_->GetDepth();

  const int grids = h / 2;
  const dim3 blocks(d / 4, 2);
  const uint8_t* src = matching_cost_->GetData();
  uint16_t* dst = cost.GetData();

  if (directions_ & DIR_LEFT_TO_RIGHT)
  {
    CUDA_LAUNCH(AggregateKernel1<128>, grids, blocks, 0, 0, src, dst,
        w, h, 20, 100);
  }

  if (directions_ & DIR_RIGHT_TO_LEFT)
  {
    CUDA_LAUNCH(AggregateKernel2<128>, grids, blocks, 0, 0, src,
        dst, w, h, 20, 100);
  }
}

void Aggregator::AggregateVertical(AggregateCost& cost) const
{
  MATCHBOX_DEBUG(matching_cost_->GetDepth() == 128);

  const int w = matching_cost_->GetWidth();
  const int h = matching_cost_->GetHeight();
  const int d = matching_cost_->GetDepth();

  const int grids = w / 2;
  const dim3 blocks(d / 4, 2);
  const uint8_t* src = matching_cost_->GetData();
  uint16_t* dst = cost.GetData();

  if (directions_ & DIR_TOP_TO_BOTTOM)
  {
    CUDA_LAUNCH(AggregateKernel3<128>, grids, blocks, 0, 0, src,
        dst, w, h, 20, 100);
  }

  if (directions_ & DIR_BOTTOM_TO_TOP)
  {
    CUDA_LAUNCH(AggregateKernel4<128>, grids, blocks, 0, 0, src,
        dst, w, h, 20, 100);
  }
}

void Aggregator::AggregateDiagonal(AggregateCost& cost) const
{
  MATCHBOX_DEBUG(matching_cost_->GetDepth() == 128);

  const int w = matching_cost_->GetWidth();
  const int h = matching_cost_->GetHeight();
  const int d = matching_cost_->GetDepth();

  const dim3 blocks(d / 4, 2);
  const int grids = (w + h) / 2;

  const uint8_t* src = matching_cost_->GetData();
  uint16_t* dst = cost.GetData();

  if (directions_ & DIR_TOP_LEFT_TO_BOTTOM_RIGHT)
  {
    CUDA_LAUNCH(AggregateKernel5<128>, grids, blocks, 0, 0, src,
        dst, w, h, 20, 100);
  }

  if (directions_ & DIR_BOTTOM_RIGHT_TO_TOP_LEFT)
  {
    CUDA_LAUNCH(AggregateKernel6<128>, grids, blocks, 0, 0, src,
        dst, w, h, 20, 100);
  }

  if (directions_ & DIR_BOTTOM_LEFT_TO_TOP_RIGHT)
  {
    CUDA_LAUNCH(AggregateKernel7<128>, grids, blocks, 0, 0, src,
        dst, w, h, 20, 100);
  }

  if (directions_ & DIR_TOP_RIGHT_TO_BOTTOM_LEFT)
  {
    CUDA_LAUNCH(AggregateKernel8<128>, grids, blocks, 0, 0, src,
        dst, w, h, 20, 100);
  }
}

void Aggregator::Initialize()
{
  for (int i = 0; i < stream_count; ++i)
  {
    CUDA_DEBUG(cudaStreamCreate(&streams[i]));
  }
}

} // namespace matchbox