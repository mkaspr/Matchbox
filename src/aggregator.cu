#include <matchbox/aggregator.h>
#include <matchbox/aggregate_cost.h>
#include <matchbox/exception.h>
#include <matchbox/matching_cost.h>
#include <matchbox/util.cuh>

namespace matchbox
{

MATCHBOX_GLOBAL
void AggregateMatchingKernel(const uint8_t* __restrict__ matching_cost,
    uint8_t* aggregrate_cost, int count)
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
    uint8_t* aggregrate_cost, int w, int h, uint8_t P1, uint8_t P2)
{
  uint32_t shared[MAX_DISP + 2];

  const int k = 4 * threadIdx.x;
  const int y = 2 * blockIdx.x + threadIdx.y;
  const uint32_t* mc = reinterpret_cast<const uint32_t*>(matching_cost);

  int aggr[] = { 0, 0, 0, 0 };
  int vmin = 0;

  const uint32_t max_cost = 64 + P2;
  shared[k + 0] = max_cost;
  shared[k + 1] = max_cost;
  shared[k + 2] = max_cost;
  shared[k + 3] = max_cost;
  shared[k + 5] = max_cost;
  shared[k + 6] = max_cost;

  for (int x = 0; x < w; ++x)
  {
    const int index = y * w * MAX_DISP + x * MAX_DISP + k;
    const uint32_t costs = mc[index >> 2];
    const uint8_t cost0 = (costs >>  0) & 0xFF;
    const uint8_t cost1 = (costs >>  8) & 0xFF;
    const uint8_t cost2 = (costs >> 16) & 0xFF;
    const uint8_t cost3 = (costs >> 24) & 0xFF;

    const uint8_t l0 = shared[k + 0] + P1;
    const uint8_t l1 = shared[k + 1] + P1;
    const uint8_t l2 = shared[k + 2] + P1;
    const uint8_t l3 = shared[k + 3] + P1;

    const uint8_t r0 = shared[k + 2] + P1;
    const uint8_t r1 = shared[k + 3] + P1;
    const uint8_t r2 = shared[k + 4] + P1;
    const uint8_t r3 = shared[k + 5] + P1;

    const uint8_t f = vmin + P2;

    aggr[0] = cost0 + min(aggr[0], min(l0, min(r0, f))) - vmin;
    aggr[1] = cost1 + min(aggr[1], min(l1, min(r1, f))) - vmin;
    aggr[2] = cost2 + min(aggr[2], min(l2, min(r2, f))) - vmin;
    aggr[3] = cost3 + min(aggr[3], min(l3, min(r3, f))) - vmin;

    aggregrate_cost[index + 0] = aggr[0];
    aggregrate_cost[index + 1] = aggr[1];
    aggregrate_cost[index + 2] = aggr[2];
    aggregrate_cost[index + 3] = aggr[3];

    shared[k + 1] = aggr[0];
    shared[k + 2] = aggr[1];
    shared[k + 3] = aggr[2];
    shared[k + 4] = aggr[3];

    const int a = min(aggr[0], min(aggr[1], min(aggr[2], aggr[3])));
    vmin = (uint8_t)WarpMin(a);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel2(const uint8_t* __restrict__ matching_cost,
    uint8_t* aggregrate_cost, int w, int h, uint8_t P1, uint8_t P2)
{
  uint32_t shared[MAX_DISP + 2];

  const int k = 4 * threadIdx.x;
  const int y = 2 * blockIdx.x + threadIdx.y;
  const uint32_t* mc = reinterpret_cast<const uint32_t*>(matching_cost);

  int aggr[] = { 0, 0, 0, 0 };
  int vmin = 0;

  const uint32_t max_cost = 64 + P2;
  shared[k + 0] = max_cost;
  shared[k + 1] = max_cost;
  shared[k + 2] = max_cost;
  shared[k + 3] = max_cost;
  shared[k + 5] = max_cost;
  shared[k + 6] = max_cost;

  for (int x = w - 1; x >= 0; --x)
  {
    const int index = y * w * MAX_DISP + x * MAX_DISP + k;
    const uint32_t costs = mc[index >> 2];
    const uint8_t cost0 = (costs >>  0) & 0xFF;
    const uint8_t cost1 = (costs >>  8) & 0xFF;
    const uint8_t cost2 = (costs >> 16) & 0xFF;
    const uint8_t cost3 = (costs >> 24) & 0xFF;

    const uint8_t l0 = shared[k + 0] + P1;
    const uint8_t l1 = shared[k + 1] + P1;
    const uint8_t l2 = shared[k + 2] + P1;
    const uint8_t l3 = shared[k + 3] + P1;

    const uint8_t r0 = shared[k + 2] + P1;
    const uint8_t r1 = shared[k + 3] + P1;
    const uint8_t r2 = shared[k + 4] + P1;
    const uint8_t r3 = shared[k + 5] + P1;

    const uint8_t f = vmin + P2;

    aggr[0] = cost0 + min(aggr[0], min(l0, min(r0, f))) - vmin;
    aggr[1] = cost1 + min(aggr[1], min(l1, min(r1, f))) - vmin;
    aggr[2] = cost2 + min(aggr[2], min(l2, min(r2, f))) - vmin;
    aggr[3] = cost3 + min(aggr[3], min(l3, min(r3, f))) - vmin;

    aggregrate_cost[index + 0] = aggr[0];
    aggregrate_cost[index + 1] = aggr[1];
    aggregrate_cost[index + 2] = aggr[2];
    aggregrate_cost[index + 3] = aggr[3];

    shared[k + 1] = aggr[0];
    shared[k + 2] = aggr[1];
    shared[k + 3] = aggr[2];
    shared[k + 4] = aggr[3];

    const int a = min(aggr[0], min(aggr[1], min(aggr[2], aggr[3])));
    vmin = (uint8_t)WarpMin(a);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel3(const uint8_t* __restrict__ matching_cost,
    uint8_t* aggregrate_cost, int w, int h, uint8_t P1, uint8_t P2)
{
  uint32_t shared[MAX_DISP + 2];

  const int k = 4 * threadIdx.x;
  const int x = 2 * blockIdx.x + threadIdx.y;
  const uint32_t* mc = reinterpret_cast<const uint32_t*>(matching_cost);

  int aggr[] = { 0, 0, 0, 0 };
  int vmin = 0;

  const uint32_t max_cost = 64 + P2;
  shared[k + 0] = max_cost;
  shared[k + 1] = max_cost;
  shared[k + 2] = max_cost;
  shared[k + 3] = max_cost;
  shared[k + 5] = max_cost;
  shared[k + 6] = max_cost;

  for (int y = 0; y < h; ++y)
  {
    const int index = y * w * MAX_DISP + x * MAX_DISP + k;
    const uint32_t costs = mc[index >> 2];
    const uint8_t cost0 = (costs >>  0) & 0xFF;
    const uint8_t cost1 = (costs >>  8) & 0xFF;
    const uint8_t cost2 = (costs >> 16) & 0xFF;
    const uint8_t cost3 = (costs >> 24) & 0xFF;

    const uint8_t l0 = shared[k + 0] + P1;
    const uint8_t l1 = shared[k + 1] + P1;
    const uint8_t l2 = shared[k + 2] + P1;
    const uint8_t l3 = shared[k + 3] + P1;

    const uint8_t r0 = shared[k + 2] + P1;
    const uint8_t r1 = shared[k + 3] + P1;
    const uint8_t r2 = shared[k + 4] + P1;
    const uint8_t r3 = shared[k + 5] + P1;

    const uint8_t f = vmin + P2;

    aggr[0] = cost0 + min(aggr[0], min(l0, min(r0, f))) - vmin;
    aggr[1] = cost1 + min(aggr[1], min(l1, min(r1, f))) - vmin;
    aggr[2] = cost2 + min(aggr[2], min(l2, min(r2, f))) - vmin;
    aggr[3] = cost3 + min(aggr[3], min(l3, min(r3, f))) - vmin;

    aggregrate_cost[index + 0] = aggr[0];
    aggregrate_cost[index + 1] = aggr[1];
    aggregrate_cost[index + 2] = aggr[2];
    aggregrate_cost[index + 3] = aggr[3];

    shared[k + 1] = aggr[0];
    shared[k + 2] = aggr[1];
    shared[k + 3] = aggr[2];
    shared[k + 4] = aggr[3];

    const int a = min(aggr[0], min(aggr[1], min(aggr[2], aggr[3])));
    vmin = (uint8_t)WarpMin(a);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel4(const uint8_t* __restrict__ matching_cost,
    uint8_t* aggregrate_cost, int w, int h, uint8_t P1, uint8_t P2)
{
  uint32_t shared[MAX_DISP + 2];

  const int k = 4 * threadIdx.x;
  const int x = 2 * blockIdx.x + threadIdx.y;
  const uint32_t* mc = reinterpret_cast<const uint32_t*>(matching_cost);

  int aggr[] = { 0, 0, 0, 0 };
  int vmin = 0;

  const uint32_t max_cost = 64 + P2;
  shared[k + 0] = max_cost;
  shared[k + 1] = max_cost;
  shared[k + 2] = max_cost;
  shared[k + 3] = max_cost;
  shared[k + 5] = max_cost;
  shared[k + 6] = max_cost;

  for (int y = h - 1; y >= 0; --y)
  {
    const int index = y * w * MAX_DISP + x * MAX_DISP + k;
    const uint32_t costs = mc[index >> 2];
    const uint8_t cost0 = (costs >>  0) & 0xFF;
    const uint8_t cost1 = (costs >>  8) & 0xFF;
    const uint8_t cost2 = (costs >> 16) & 0xFF;
    const uint8_t cost3 = (costs >> 24) & 0xFF;

    const uint8_t l0 = shared[k + 0] + P1;
    const uint8_t l1 = shared[k + 1] + P1;
    const uint8_t l2 = shared[k + 2] + P1;
    const uint8_t l3 = shared[k + 3] + P1;

    const uint8_t r0 = shared[k + 2] + P1;
    const uint8_t r1 = shared[k + 3] + P1;
    const uint8_t r2 = shared[k + 4] + P1;
    const uint8_t r3 = shared[k + 5] + P1;

    const uint8_t f = vmin + P2;

    aggr[0] = cost0 + min(aggr[0], min(l0, min(r0, f))) - vmin;
    aggr[1] = cost1 + min(aggr[1], min(l1, min(r1, f))) - vmin;
    aggr[2] = cost2 + min(aggr[2], min(l2, min(r2, f))) - vmin;
    aggr[3] = cost3 + min(aggr[3], min(l3, min(r3, f))) - vmin;

    aggregrate_cost[index + 0] = aggr[0];
    aggregrate_cost[index + 1] = aggr[1];
    aggregrate_cost[index + 2] = aggr[2];
    aggregrate_cost[index + 3] = aggr[3];

    shared[k + 1] = aggr[0];
    shared[k + 2] = aggr[1];
    shared[k + 3] = aggr[2];
    shared[k + 4] = aggr[3];

    const int a = min(aggr[0], min(aggr[1], min(aggr[2], aggr[3])));
    vmin = (uint8_t)WarpMin(a);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel5(const uint8_t* __restrict__ matching_cost,
    uint8_t* aggregrate_cost, int w, int h, uint8_t P1, uint8_t P2)
{
  uint32_t shared[MAX_DISP + 2];

  const int k = 4 * threadIdx.x;
  const int ii = 2 * blockIdx.x + threadIdx.y;
  if (ii >= h + w) return;

  const int x = (ii < h) ? 0 : ii - h + 1;
  const int y = (ii < h) ? h - ii - 1 : 0;
  const int n = min(w - x, h - y);

  const uint32_t* mc = reinterpret_cast<const uint32_t*>(matching_cost);

  int aggr[] = { 0, 0, 0, 0 };
  int vmin = 0;

  const uint32_t max_cost = 64 + P2;
  shared[k + 0] = max_cost;
  shared[k + 1] = max_cost;
  shared[k + 2] = max_cost;
  shared[k + 3] = max_cost;
  shared[k + 5] = max_cost;
  shared[k + 6] = max_cost;

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

    const uint8_t l0 = shared[k + 0] + P1;
    const uint8_t l1 = shared[k + 1] + P1;
    const uint8_t l2 = shared[k + 2] + P1;
    const uint8_t l3 = shared[k + 3] + P1;

    const uint8_t r0 = shared[k + 2] + P1;
    const uint8_t r1 = shared[k + 3] + P1;
    const uint8_t r2 = shared[k + 4] + P1;
    const uint8_t r3 = shared[k + 5] + P1;

    const uint8_t f = vmin + P2;

    aggr[0] = cost0 + min(aggr[0], min(l0, min(r0, f))) - vmin;
    aggr[1] = cost1 + min(aggr[1], min(l1, min(r1, f))) - vmin;
    aggr[2] = cost2 + min(aggr[2], min(l2, min(r2, f))) - vmin;
    aggr[3] = cost3 + min(aggr[3], min(l3, min(r3, f))) - vmin;

    aggregrate_cost[index + 0] = aggr[0];
    aggregrate_cost[index + 1] = aggr[1];
    aggregrate_cost[index + 2] = aggr[2];
    aggregrate_cost[index + 3] = aggr[3];

    shared[k + 1] = aggr[0];
    shared[k + 2] = aggr[1];
    shared[k + 3] = aggr[2];
    shared[k + 4] = aggr[3];

    const int a = min(aggr[0], min(aggr[1], min(aggr[2], aggr[3])));
    vmin = (uint8_t)WarpMin(a);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel6(const uint8_t* __restrict__ matching_cost,
    uint8_t* aggregrate_cost, int w, int h, uint8_t P1, uint8_t P2)
{
  uint32_t shared[MAX_DISP + 2];

  const int k = 4 * threadIdx.x;
  const int ii = 2 * blockIdx.x + threadIdx.y;
  if (ii >= h + w) return;

  const int x = (ii < h) ? 0 : ii - h + 1;
  const int y = (ii < h) ? h - ii - 1 : 0;
  const int n = min(w - x, h - y);

  const uint32_t* mc = reinterpret_cast<const uint32_t*>(matching_cost);

  int aggr[] = { 0, 0, 0, 0 };
  int vmin = 0;

  const uint32_t max_cost = 64 + P2;
  shared[k + 0] = max_cost;
  shared[k + 1] = max_cost;
  shared[k + 2] = max_cost;
  shared[k + 3] = max_cost;
  shared[k + 5] = max_cost;
  shared[k + 6] = max_cost;

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

    const uint8_t l0 = shared[k + 0] + P1;
    const uint8_t l1 = shared[k + 1] + P1;
    const uint8_t l2 = shared[k + 2] + P1;
    const uint8_t l3 = shared[k + 3] + P1;

    const uint8_t r0 = shared[k + 2] + P1;
    const uint8_t r1 = shared[k + 3] + P1;
    const uint8_t r2 = shared[k + 4] + P1;
    const uint8_t r3 = shared[k + 5] + P1;

    const uint8_t f = vmin + P2;

    aggr[0] = cost0 + min(aggr[0], min(l0, min(r0, f))) - vmin;
    aggr[1] = cost1 + min(aggr[1], min(l1, min(r1, f))) - vmin;
    aggr[2] = cost2 + min(aggr[2], min(l2, min(r2, f))) - vmin;
    aggr[3] = cost3 + min(aggr[3], min(l3, min(r3, f))) - vmin;

    aggregrate_cost[index + 0] = aggr[0];
    aggregrate_cost[index + 1] = aggr[1];
    aggregrate_cost[index + 2] = aggr[2];
    aggregrate_cost[index + 3] = aggr[3];

    shared[k + 1] = aggr[0];
    shared[k + 2] = aggr[1];
    shared[k + 3] = aggr[2];
    shared[k + 4] = aggr[3];

    const int a = min(aggr[0], min(aggr[1], min(aggr[2], aggr[3])));
    vmin = (uint8_t)WarpMin(a);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel7(const uint8_t* __restrict__ matching_cost,
    uint8_t* aggregrate_cost, int w, int h, uint8_t P1, uint8_t P2)
{
  uint32_t shared[MAX_DISP + 2];

  const int k = 4 * threadIdx.x;
  const int ii = 2 * blockIdx.x + threadIdx.y;
  if (ii >= h + w) return;

  const int x = (ii < h) ? 0 : ii - h + 1;
  const int y = (ii < h) ? h - ii - 1 : 0;
  const int n = min(w - x, h - y);

  const uint32_t* mc = reinterpret_cast<const uint32_t*>(matching_cost);

  int aggr[] = { 0, 0, 0, 0 };
  int vmin = 0;

  const uint32_t max_cost = 64 + P2;
  shared[k + 0] = max_cost;
  shared[k + 1] = max_cost;
  shared[k + 2] = max_cost;
  shared[k + 3] = max_cost;
  shared[k + 5] = max_cost;
  shared[k + 6] = max_cost;

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

    const uint8_t l0 = shared[k + 0] + P1;
    const uint8_t l1 = shared[k + 1] + P1;
    const uint8_t l2 = shared[k + 2] + P1;
    const uint8_t l3 = shared[k + 3] + P1;

    const uint8_t r0 = shared[k + 2] + P1;
    const uint8_t r1 = shared[k + 3] + P1;
    const uint8_t r2 = shared[k + 4] + P1;
    const uint8_t r3 = shared[k + 5] + P1;

    const uint8_t f = vmin + P2;

    aggr[0] = cost0 + min(aggr[0], min(l0, min(r0, f))) - vmin;
    aggr[1] = cost1 + min(aggr[1], min(l1, min(r1, f))) - vmin;
    aggr[2] = cost2 + min(aggr[2], min(l2, min(r2, f))) - vmin;
    aggr[3] = cost3 + min(aggr[3], min(l3, min(r3, f))) - vmin;

    aggregrate_cost[index + 0] = aggr[0];
    aggregrate_cost[index + 1] = aggr[1];
    aggregrate_cost[index + 2] = aggr[2];
    aggregrate_cost[index + 3] = aggr[3];

    shared[k + 1] = aggr[0];
    shared[k + 2] = aggr[1];
    shared[k + 3] = aggr[2];
    shared[k + 4] = aggr[3];

    const int a = min(aggr[0], min(aggr[1], min(aggr[2], aggr[3])));
    vmin = (uint8_t)WarpMin(a);
  }
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void AggregateKernel8(const uint8_t* __restrict__ matching_cost,
    uint8_t* aggregrate_cost, int w, int h, uint8_t P1, uint8_t P2)
{
  uint32_t shared[MAX_DISP + 2];

  const int k = 4 * threadIdx.x;
  const int ii = 2 * blockIdx.x + threadIdx.y;
  if (ii >= h + w) return;

  const int x = (ii < h) ? 0 : ii - h + 1;
  const int y = (ii < h) ? h - ii - 1 : 0;
  const int n = min(w - x, h - y);

  const uint32_t* mc = reinterpret_cast<const uint32_t*>(matching_cost);

  int aggr[] = { 0, 0, 0, 0 };
  int vmin = 0;

  const uint32_t max_cost = 64 + P2;
  shared[k + 0] = max_cost;
  shared[k + 1] = max_cost;
  shared[k + 2] = max_cost;
  shared[k + 3] = max_cost;
  shared[k + 5] = max_cost;
  shared[k + 6] = max_cost;

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

    const uint8_t l0 = shared[k + 0] + P1;
    const uint8_t l1 = shared[k + 1] + P1;
    const uint8_t l2 = shared[k + 2] + P1;
    const uint8_t l3 = shared[k + 3] + P1;

    const uint8_t r0 = shared[k + 2] + P1;
    const uint8_t r1 = shared[k + 3] + P1;
    const uint8_t r2 = shared[k + 4] + P1;
    const uint8_t r3 = shared[k + 5] + P1;

    const uint8_t f = vmin + P2;

    aggr[0] = cost0 + min(aggr[0], min(l0, min(r0, f))) - vmin;
    aggr[1] = cost1 + min(aggr[1], min(l1, min(r1, f))) - vmin;
    aggr[2] = cost2 + min(aggr[2], min(l2, min(r2, f))) - vmin;
    aggr[3] = cost3 + min(aggr[3], min(l3, min(r3, f))) - vmin;

    aggregrate_cost[index + 0] = aggr[0];
    aggregrate_cost[index + 1] = aggr[1];
    aggregrate_cost[index + 2] = aggr[2];
    aggregrate_cost[index + 3] = aggr[3];

    shared[k + 1] = aggr[0];
    shared[k + 2] = aggr[1];
    shared[k + 3] = aggr[2];
    shared[k + 4] = aggr[3];

    const int a = min(aggr[0], min(aggr[1], min(aggr[2], aggr[3])));
    vmin = (uint8_t)WarpMin(a);
  }
}

Aggregator::Aggregator(std::shared_ptr<const MatchingCost> matching_cost) :
  matching_cost_(matching_cost),
  degree_(3)
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
  uint8_t* dst = cost.GetData();

  CUDA_LAUNCH(AggregateMatchingKernel, grids, blocks, 0, 0, src, dst, threads);
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
  uint8_t* dst = cost.GetData();

  const int offset = w * h * d;

  CUDA_LAUNCH(AggregateKernel<128>, grids, blocks, 0, streams[0], src, dst,
      w, h, 20, 100);

  CUDA_LAUNCH(AggregateKernel2<128>, grids, blocks, 0, streams[1], src,
      dst + offset, w, h, 20, 100);
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
  uint8_t* dst = cost.GetData();

  int offset = 2 * w * h * d;

  CUDA_LAUNCH(AggregateKernel3<128>, grids, blocks, 0, streams[2], src,
      dst + offset, w, h, 20, 100);

  offset = 3 * w * h * d;

  CUDA_LAUNCH(AggregateKernel4<128>, grids, blocks, 0, streams[3], src,
      dst + offset, w, h, 20, 100);
}

void Aggregator::AggregateDiagonal(AggregateCost& cost) const
{
  MATCHBOX_DEBUG(matching_cost_->GetDepth() == 128);

  const int w = matching_cost_->GetWidth();
  const int h = matching_cost_->GetHeight();
  const int d = matching_cost_->GetDepth();

  const dim3 blocks(d / 4, 2);
  const int grids = (w + h) / 2; // assuming even dims

  const uint8_t* src = matching_cost_->GetData();
  uint8_t* dst = cost.GetData();

  int offset = 4 * w * h * d;

  CUDA_LAUNCH(AggregateKernel5<128>, grids, blocks, 0, streams[0], src,
      dst + offset, w, h, 20, 100);

  offset = 5 * w * h * d;

  CUDA_LAUNCH(AggregateKernel6<128>, grids, blocks, 0, streams[1], src,
      dst + offset, w, h, 20, 100);

  offset = 6 * w * h * d;

  CUDA_LAUNCH(AggregateKernel7<128>, grids, blocks, 0, streams[2], src,
      dst + offset, w, h, 20, 100);

  offset = 7 * w * h * d;

  CUDA_LAUNCH(AggregateKernel8<128>, grids, blocks, 0, streams[3], src,
      dst + offset, w, h, 20, 100);
}

void Aggregator::Initialize()
{
  for (int i = 0; i < stream_count; ++i)
  {
    CUDA_DEBUG(cudaStreamCreate(&streams[i]));
  }
}

} // namespace matchbox