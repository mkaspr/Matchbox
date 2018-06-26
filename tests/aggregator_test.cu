#include <gtest/gtest.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <matchbox/aggregate_cost.h>
#include <matchbox/aggregator.h>
#include <matchbox/matching_cost.h>

namespace matchbox
{
namespace testing
{
namespace
{

inline std::shared_ptr<MatchingCost> CreateMatchingCost()
{
  const int w = 64;
  const int h = 48;
  const int d = 128;

  std::shared_ptr<MatchingCost> cost;
  cost = std::make_shared<MatchingCost>(w, h, d);
  thrust::host_vector<uint16_t> data(cost->GetTotal());
  int index = 0;

  for (int i = 0; i < h; ++i)
  {
    for (int j = 0; j < w; ++j)
    {
      for (int k = 0; k < d; ++k)
      {
        data[index++] = i % 31 + j % 17 + k % 7;
      }
    }
  }

  thrust::device_ptr<uint8_t> ptr(cost->GetData());
  thrust::copy(data.begin(), data.end(), ptr);
  return cost;
}

inline void AggregateLeftToRight(std::shared_ptr<const MatchingCost> src,
    std::vector<uint16_t>& dst)
{
  const int w = src->GetWidth();
  const int h = src->GetHeight();
  const int m = src->GetDepth();

  dst.resize(src->GetTotal());
  thrust::device_ptr<const uint8_t> ptr(src->GetData());
  thrust::host_vector<uint8_t> data(ptr, ptr + src->GetTotal());

  std::vector<int> aggrs(m);
  std::vector<int> curr_shared(m + 2);
  std::vector<int> next_shared(m + 2);

  for (int y = 0; y < h; ++y)
  {
    std::fill(aggrs.begin(), aggrs.end(), 0);
    std::fill(curr_shared.begin(), curr_shared.end(), 64 + 100);
    std::fill(next_shared.begin(), next_shared.end(), 64 + 100);
    int curr_min = 0;

    for (int x = 0; x < w; ++x)
    {
      int next_min = 100000;

      for (int d = 0; d < m; ++d)
      {
        const int index = y * w * m + x * m + d;
        int matching_cost  = data[index];
        int left  = curr_shared[d];
        int right = curr_shared[d + 2];

        const int a = aggrs[d];
        const int b = left + 20;
        const int c = right + 20;
        const int e = curr_min + 100;
        const int delta = std::min(a, std::min(b, std::min(c, e)));

        aggrs[d] = matching_cost + delta - curr_min;
        dst[index] += aggrs[d];
        next_shared[d + 1] = aggrs[d];

        if (aggrs[d] < next_min) next_min = aggrs[d];
      }

      curr_min = next_min;
      std::swap(curr_shared, next_shared);
    }
  }
}

inline void AggregateRightToLeft(std::shared_ptr<const MatchingCost> src,
    std::vector<uint16_t>& dst)
{
  const int w = src->GetWidth();
  const int h = src->GetHeight();
  const int m = src->GetDepth();

  dst.resize(src->GetTotal());
  thrust::device_ptr<const uint8_t> ptr(src->GetData());
  thrust::host_vector<uint8_t> data(ptr, ptr + src->GetTotal());

  std::vector<int> aggrs(m);
  std::vector<int> curr_shared(m + 2);
  std::vector<int> next_shared(m + 2);

  for (int y = 0; y < h; ++y)
  {
    std::fill(aggrs.begin(), aggrs.end(), 0);
    std::fill(curr_shared.begin(), curr_shared.end(), 64 + 100);
    std::fill(next_shared.begin(), next_shared.end(), 64 + 100);
    int curr_min = 0;

    for (int x = w - 1; x >= 0; --x)
    {
      int next_min = 100000;

      for (int d = 0; d < m; ++d)
      {
        const int index = y * w * m + x * m + d;
        int matching_cost  = data[y * w * m + x * m + d];
        int left  = curr_shared[d];
        int right = curr_shared[d + 2];

        const int a = aggrs[d];
        const int b = left + 20;
        const int c = right + 20;
        const int e = curr_min + 100;
        const int delta = std::min(a, std::min(b, std::min(c, e)));

        aggrs[d] = matching_cost + delta - curr_min;
        dst[index] += aggrs[d];
        next_shared[d + 1] = aggrs[d];

        if (aggrs[d] < next_min) next_min = aggrs[d];
      }

      curr_min = next_min;
      std::swap(curr_shared, next_shared);
    }
  }
}

inline void AggregateTopToBottom(std::shared_ptr<const MatchingCost> src,
    std::vector<uint16_t>& dst)
{
  const int w = src->GetWidth();
  const int h = src->GetHeight();
  const int m = src->GetDepth();

  dst.resize(src->GetTotal());
  thrust::device_ptr<const uint8_t> ptr(src->GetData());
  thrust::host_vector<uint8_t> data(ptr, ptr + src->GetTotal());

  std::vector<int> aggrs(m);
  std::vector<int> curr_shared(m + 2);
  std::vector<int> next_shared(m + 2);

  for (int x = 0; x < w; ++x)
  {
    std::fill(aggrs.begin(), aggrs.end(), 0);
    std::fill(curr_shared.begin(), curr_shared.end(), 64 + 100);
    std::fill(next_shared.begin(), next_shared.end(), 64 + 100);
    int curr_min = 0;

    for (int y = 0; y < h; ++y)
    {
      int next_min = 100000;

      for (int d = 0; d < m; ++d)
      {
        const int index = y * w * m + x * m + d;
        int matching_cost  = data[index];
        int left  = curr_shared[d];
        int right = curr_shared[d + 2];

        const int a = aggrs[d];
        const int b = left + 20;
        const int c = right + 20;
        const int e = curr_min + 100;
        const int delta = std::min(a, std::min(b, std::min(c, e)));

        aggrs[d] = matching_cost + delta - curr_min;
        dst[index] += aggrs[d];
        next_shared[d + 1] = aggrs[d];

        if (aggrs[d] < next_min) next_min = aggrs[d];
      }

      curr_min = next_min;
      std::swap(curr_shared, next_shared);
    }
  }
}

inline void AggregateBottomToTop(std::shared_ptr<const MatchingCost> src,
    std::vector<uint16_t>& dst)
{
  const int w = src->GetWidth();
  const int h = src->GetHeight();
  const int m = src->GetDepth();

  dst.resize(src->GetTotal());
  thrust::device_ptr<const uint8_t> ptr(src->GetData());
  thrust::host_vector<uint8_t> data(ptr, ptr + src->GetTotal());

  std::vector<int> aggrs(m);
  std::vector<int> curr_shared(m + 2);
  std::vector<int> next_shared(m + 2);

  for (int x = 0; x < w; ++x)
  {
    std::fill(aggrs.begin(), aggrs.end(), 0);
    std::fill(curr_shared.begin(), curr_shared.end(), 64 + 100);
    std::fill(next_shared.begin(), next_shared.end(), 64 + 100);
    int curr_min = 0;

    for (int y = h - 1; y >= 0; --y)
    {
      int next_min = 100000;

      for (int d = 0; d < m; ++d)
      {
        const int index = y * w * m + x * m + d;
        int matching_cost  = data[index];
        int left  = curr_shared[d];
        int right = curr_shared[d + 2];

        const int a = aggrs[d];
        const int b = left + 20;
        const int c = right + 20;
        const int e = curr_min + 100;
        const int delta = std::min(a, std::min(b, std::min(c, e)));

        aggrs[d] = matching_cost + delta - curr_min;
        dst[index] += aggrs[d];
        next_shared[d + 1] = aggrs[d];

        if (aggrs[d] < next_min) next_min = aggrs[d];
      }

      curr_min = next_min;
      std::swap(curr_shared, next_shared);
    }
  }
}

inline void AggregateTopLeftToBottomRight(
    std::shared_ptr<const MatchingCost> src, std::vector<uint16_t>& dst)
{
  const int w = src->GetWidth();
  const int h = src->GetHeight();
  const int m = src->GetDepth();

  dst.resize(src->GetTotal());
  thrust::device_ptr<const uint8_t> ptr(src->GetData());
  thrust::host_vector<uint8_t> data(ptr, ptr + src->GetTotal());

  std::vector<int> aggrs(m);
  std::vector<int> curr_shared(m + 2);
  std::vector<int> next_shared(m + 2);

  int n = w + h - 1;

  for (int i = 0; i < n; ++i)
  {
    const int x = (i < h) ? 0 : i - h + 1;
    const int y = (i < h) ? h - i - 1 : 0;
    const int q = std::min(w - x, h - y);

    std::fill(aggrs.begin(), aggrs.end(), 0);
    std::fill(curr_shared.begin(), curr_shared.end(), 64 + 100);
    std::fill(next_shared.begin(), next_shared.end(), 64 + 100);
    int curr_min = 0;

    for (int j = 0; j < q; ++j)
    {
      int next_min = 100000;

      for (int d = 0; d < m; ++d)
      {
        const int index = (y + j) * w * m + (x + j) * m + d;
        int matching_cost  = data[index];
        int left  = curr_shared[d];
        int right = curr_shared[d + 2];

        const int a = aggrs[d];
        const int b = left + 20;
        const int c = right + 20;
        const int e = curr_min + 100;
        const int delta = std::min(a, std::min(b, std::min(c, e)));
        aggrs[d] = matching_cost + delta - curr_min;
        dst[index] += aggrs[d];
        next_shared[d + 1] = aggrs[d];

        if (aggrs[d] < next_min) next_min = aggrs[d];
      }

      curr_min = next_min;
      std::swap(curr_shared, next_shared);
    }
  }
}

inline void AggregateBottomRightToTopLeft(
    std::shared_ptr<const MatchingCost> src, std::vector<uint16_t>& dst)
{
  const int w = src->GetWidth();
  const int h = src->GetHeight();
  const int m = src->GetDepth();

  dst.resize(src->GetTotal());
  thrust::device_ptr<const uint8_t> ptr(src->GetData());
  thrust::host_vector<uint8_t> data(ptr, ptr + src->GetTotal());

  std::vector<int> aggrs(m);
  std::vector<int> curr_shared(m + 2);
  std::vector<int> next_shared(m + 2);

  int n = w + h - 1;

  for (int i = 0; i < n; ++i)
  {
    const int x = (i < h) ? 0 : i - h + 1;
    const int y = (i < h) ? h - i - 1 : 0;
    const int q = std::min(w - x, h - y);

    std::fill(aggrs.begin(), aggrs.end(), 0);
    std::fill(curr_shared.begin(), curr_shared.end(), 64 + 100);
    std::fill(next_shared.begin(), next_shared.end(), 64 + 100);
    int curr_min = 0;

    for (int j = 0; j < q; ++j)
    {
      int next_min = 100000;

      for (int d = 0; d < m; ++d)
      {
        const int xx = w - x - j - 1;
        const int yy = h - y - j - 1;
        const int index = yy * w * m + xx * m + d;
        int matching_cost  = data[index];
        int left  = curr_shared[d];
        int right = curr_shared[d + 2];

        const int a = aggrs[d];
        const int b = left + 20;
        const int c = right + 20;
        const int e = curr_min + 100;
        const int delta = std::min(a, std::min(b, std::min(c, e)));
        aggrs[d] = matching_cost + delta - curr_min;
        dst[index] += aggrs[d];
        next_shared[d + 1] = aggrs[d];

        if (aggrs[d] < next_min) next_min = aggrs[d];
      }

      curr_min = next_min;
      std::swap(curr_shared, next_shared);
    }
  }
}

inline void AggregateBottomLeftToTopRight(
    std::shared_ptr<const MatchingCost> src, std::vector<uint16_t>& dst)
{
  const int w = src->GetWidth();
  const int h = src->GetHeight();
  const int m = src->GetDepth();

  dst.resize(src->GetTotal());
  thrust::device_ptr<const uint8_t> ptr(src->GetData());
  thrust::host_vector<uint8_t> data(ptr, ptr + src->GetTotal());

  std::vector<int> aggrs(m);
  std::vector<int> curr_shared(m + 2);
  std::vector<int> next_shared(m + 2);

  int n = w + h - 1;

  for (int i = 0; i < n; ++i)
  {
    const int x = (i < h) ? 0 : i - h + 1;
    const int y = (i < h) ? h - i - 1 : 0;
    const int q = std::min(w - x, h - y);

    std::fill(aggrs.begin(), aggrs.end(), 0);
    std::fill(curr_shared.begin(), curr_shared.end(), 64 + 100);
    std::fill(next_shared.begin(), next_shared.end(), 64 + 100);
    int curr_min = 0;

    for (int j = 0; j < q; ++j)
    {
      int next_min = 100000;

      for (int d = 0; d < m; ++d)
      {
        const int xx = x + j;
        const int yy = h - y - j - 1;
        const int index = yy * w * m + xx * m + d;
        int matching_cost  = data[index];
        int left  = curr_shared[d];
        int right = curr_shared[d + 2];

        const int a = aggrs[d];
        const int b = left + 20;
        const int c = right + 20;
        const int e = curr_min + 100;
        const int delta = std::min(a, std::min(b, std::min(c, e)));
        aggrs[d] = matching_cost + delta - curr_min;
        dst[index] += aggrs[d];
        next_shared[d + 1] = aggrs[d];

        if (aggrs[d] < next_min) next_min = aggrs[d];
      }

      curr_min = next_min;
      std::swap(curr_shared, next_shared);
    }
  }
}

inline void AggregateTopRightToBottomLeft(
    std::shared_ptr<const MatchingCost> src, std::vector<uint16_t>& dst)
{
  const int w = src->GetWidth();
  const int h = src->GetHeight();
  const int m = src->GetDepth();

  dst.resize(src->GetTotal());
  thrust::device_ptr<const uint8_t> ptr(src->GetData());
  thrust::host_vector<uint8_t> data(ptr, ptr + src->GetTotal());

  std::vector<int> aggrs(m);
  std::vector<int> curr_shared(m + 2);
  std::vector<int> next_shared(m + 2);

  int n = w + h - 1;

  for (int i = 0; i < n; ++i)
  {
    const int x = (i < h) ? 0 : i - h + 1;
    const int y = (i < h) ? h - i - 1 : 0;
    const int q = std::min(w - x, h - y);

    std::fill(aggrs.begin(), aggrs.end(), 0);
    std::fill(curr_shared.begin(), curr_shared.end(), 64 + 100);
    std::fill(next_shared.begin(), next_shared.end(), 64 + 100);
    int curr_min = 0;

    for (int j = 0; j < q; ++j)
    {
      int next_min = 100000;

      for (int d = 0; d < m; ++d)
      {
        const int xx = w - x - j - 1;
        const int yy = y + j;
        const int index = yy * w * m + xx * m + d;
        int matching_cost  = data[index];
        int left  = curr_shared[d];
        int right = curr_shared[d + 2];

        const int a = aggrs[d];
        const int b = left + 20;
        const int c = right + 20;
        const int e = curr_min + 100;
        const int delta = std::min(a, std::min(b, std::min(c, e)));
        aggrs[d] = matching_cost + delta - curr_min;
        dst[index] += aggrs[d];
        next_shared[d + 1] = aggrs[d];

        if (aggrs[d] < next_min) next_min = aggrs[d];
      }

      curr_min = next_min;
      std::swap(curr_shared, next_shared);
    }
  }
}

} // namespace

TEST(Aggregator, Constructor)
{
  std::shared_ptr<MatchingCost> matching_cost;
  matching_cost = std::make_shared<MatchingCost>();
  Aggregator aggregator(matching_cost);
  ASSERT_EQ(matching_cost, aggregator.GetMatchingCost());
  ASSERT_EQ(Aggregator::DIR_ALL, aggregator.GetDirections());
}

TEST(Aggregator, AggregateNone)
{
  std::shared_ptr<MatchingCost> matching_cost = CreateMatchingCost();
  Aggregator aggregator(matching_cost);
  aggregator.SetDirections(Aggregator::DIR_NONE);
  AggregateCost aggregate_cost;
  aggregator.Aggregate(aggregate_cost);

  ASSERT_EQ(matching_cost->GetWidth(),  aggregate_cost.GetWidth());
  ASSERT_EQ(matching_cost->GetHeight(), aggregate_cost.GetHeight());
  ASSERT_EQ(matching_cost->GetDepth(),  aggregate_cost.GetDepth());

  const int count = matching_cost->GetTotal();
  thrust::device_ptr<const uint8_t> expected_ptr(matching_cost->GetData());
  thrust::device_ptr<const uint16_t> found_ptr(aggregate_cost.GetData());
  thrust::host_vector<uint8_t> expected(expected_ptr, expected_ptr + count);
  thrust::host_vector<uint16_t> found(found_ptr, found_ptr + count);

  for (int i = 0; i < (int)expected.size(); ++i)
  {
    ASSERT_EQ(expected[i], found[i]);
  }
}

TEST(Aggregator, AggregateLeftToRight)
{
  std::shared_ptr<MatchingCost> matching_cost = CreateMatchingCost();

  std::vector<uint16_t> expected(matching_cost->GetTotal());
  std::fill(expected.begin(), expected.end(), 0);
  AggregateLeftToRight(matching_cost, expected);

  Aggregator aggregator(matching_cost);
  aggregator.SetDirections(Aggregator::DIR_LEFT_TO_RIGHT);
  AggregateCost aggregate_cost;
  aggregator.Aggregate(aggregate_cost);

  ASSERT_EQ(matching_cost->GetWidth(),  aggregate_cost.GetWidth());
  ASSERT_EQ(matching_cost->GetHeight(), aggregate_cost.GetHeight());
  ASSERT_EQ(matching_cost->GetDepth(),  aggregate_cost.GetDepth());

  const int count = matching_cost->GetTotal();
  thrust::device_ptr<const uint16_t> found_ptr(aggregate_cost.GetData());
  thrust::host_vector<uint16_t> found(found_ptr, found_ptr + count);

  for (int i = 0; i < (int)expected.size(); ++i)
  {
    ASSERT_EQ(expected[i], found[i]);
  }
}

TEST(Aggregator, AggregateRightToLeft)
{
  std::shared_ptr<MatchingCost> matching_cost = CreateMatchingCost();

  std::vector<uint16_t> expected(matching_cost->GetTotal());
  std::fill(expected.begin(), expected.end(), 0);
  AggregateRightToLeft(matching_cost, expected);

  Aggregator aggregator(matching_cost);
  aggregator.SetDirections(Aggregator::DIR_RIGHT_TO_LEFT);
  AggregateCost aggregate_cost;
  aggregator.Aggregate(aggregate_cost);

  ASSERT_EQ(matching_cost->GetWidth(),  aggregate_cost.GetWidth());
  ASSERT_EQ(matching_cost->GetHeight(), aggregate_cost.GetHeight());
  ASSERT_EQ(matching_cost->GetDepth(),  aggregate_cost.GetDepth());

  const int count = matching_cost->GetTotal();
  thrust::device_ptr<const uint16_t> found_ptr(aggregate_cost.GetData());
  thrust::host_vector<uint16_t> found(found_ptr, found_ptr + count);

  for (int i = 0; i < (int)expected.size(); ++i)
  {
    ASSERT_EQ(expected[i], found[i]);
  }
}

TEST(Aggregator, AggregateTopToBottom)
{
  std::shared_ptr<MatchingCost> matching_cost = CreateMatchingCost();

  std::vector<uint16_t> expected(matching_cost->GetTotal());
  std::fill(expected.begin(), expected.end(), 0);
  AggregateTopToBottom(matching_cost, expected);

  Aggregator aggregator(matching_cost);
  aggregator.SetDirections(Aggregator::DIR_TOP_TO_BOTTOM);
  AggregateCost aggregate_cost;
  aggregator.Aggregate(aggregate_cost);

  ASSERT_EQ(matching_cost->GetWidth(),  aggregate_cost.GetWidth());
  ASSERT_EQ(matching_cost->GetHeight(), aggregate_cost.GetHeight());
  ASSERT_EQ(matching_cost->GetDepth(),  aggregate_cost.GetDepth());

  const int count = matching_cost->GetTotal();
  thrust::device_ptr<const uint16_t> found_ptr(aggregate_cost.GetData());
  thrust::host_vector<uint16_t> found(found_ptr, found_ptr + count);

  for (int i = 0; i < (int)expected.size(); ++i)
  {
    ASSERT_EQ(expected[i], found[i]);
  }
}

TEST(Aggregator, AggregateBottomToTop)
{
  std::shared_ptr<MatchingCost> matching_cost = CreateMatchingCost();

  std::vector<uint16_t> expected(matching_cost->GetTotal());
  std::fill(expected.begin(), expected.end(), 0);
  AggregateBottomToTop(matching_cost, expected);

  Aggregator aggregator(matching_cost);
  aggregator.SetDirections(Aggregator::DIR_BOTTOM_TO_TOP);
  AggregateCost aggregate_cost;
  aggregator.Aggregate(aggregate_cost);

  ASSERT_EQ(matching_cost->GetWidth(),  aggregate_cost.GetWidth());
  ASSERT_EQ(matching_cost->GetHeight(), aggregate_cost.GetHeight());
  ASSERT_EQ(matching_cost->GetDepth(),  aggregate_cost.GetDepth());

  const int count = matching_cost->GetTotal();
  thrust::device_ptr<const uint16_t> found_ptr(aggregate_cost.GetData());
  thrust::host_vector<uint16_t> found(found_ptr, found_ptr + count);

  for (int i = 0; i < (int)expected.size(); ++i)
  {
    ASSERT_EQ(expected[i], found[i]);
  }
}

TEST(Aggregator, AggregateTopLeftToBottomRight)
{
  std::shared_ptr<MatchingCost> matching_cost = CreateMatchingCost();

  std::vector<uint16_t> expected(matching_cost->GetTotal());
  std::fill(expected.begin(), expected.end(), 0);
  AggregateTopLeftToBottomRight(matching_cost, expected);

  Aggregator aggregator(matching_cost);
  aggregator.SetDirections(Aggregator::DIR_TOP_LEFT_TO_BOTTOM_RIGHT);
  AggregateCost aggregate_cost;
  aggregator.Aggregate(aggregate_cost);

  ASSERT_EQ(matching_cost->GetWidth(),  aggregate_cost.GetWidth());
  ASSERT_EQ(matching_cost->GetHeight(), aggregate_cost.GetHeight());
  ASSERT_EQ(matching_cost->GetDepth(),  aggregate_cost.GetDepth());

  const int count = matching_cost->GetTotal();
  thrust::device_ptr<const uint16_t> found_ptr(aggregate_cost.GetData());
  thrust::host_vector<uint16_t> found(found_ptr, found_ptr + count);

  for (int i = 0; i < (int)expected.size(); ++i)
  {
    ASSERT_EQ(expected[i], found[i]);
  }
}

TEST(Aggregator, AggregateBottomRightToTopLeft)
{
  std::shared_ptr<MatchingCost> matching_cost = CreateMatchingCost();

  std::vector<uint16_t> expected(matching_cost->GetTotal());
  std::fill(expected.begin(), expected.end(), 0);
  AggregateBottomRightToTopLeft(matching_cost, expected);

  Aggregator aggregator(matching_cost);
  aggregator.SetDirections(Aggregator::DIR_BOTTOM_RIGHT_TO_TOP_LEFT);
  AggregateCost aggregate_cost;
  aggregator.Aggregate(aggregate_cost);

  ASSERT_EQ(matching_cost->GetWidth(),  aggregate_cost.GetWidth());
  ASSERT_EQ(matching_cost->GetHeight(), aggregate_cost.GetHeight());
  ASSERT_EQ(matching_cost->GetDepth(),  aggregate_cost.GetDepth());

  const int count = matching_cost->GetTotal();
  thrust::device_ptr<const uint16_t> found_ptr(aggregate_cost.GetData());
  thrust::host_vector<uint16_t> found(found_ptr, found_ptr + count);

  for (int i = 0; i < (int)expected.size(); ++i)
  {
    ASSERT_EQ(expected[i], found[i]);
  }
}

TEST(Aggregator, AggregateBottomLeftToTopRight)
{
  std::shared_ptr<MatchingCost> matching_cost = CreateMatchingCost();

  std::vector<uint16_t> expected(matching_cost->GetTotal());
  std::fill(expected.begin(), expected.end(), 0);
  AggregateBottomLeftToTopRight(matching_cost, expected);

  Aggregator aggregator(matching_cost);
  aggregator.SetDirections(Aggregator::DIR_BOTTOM_LEFT_TO_TOP_RIGHT);
  AggregateCost aggregate_cost;
  aggregator.Aggregate(aggregate_cost);

  ASSERT_EQ(matching_cost->GetWidth(),  aggregate_cost.GetWidth());
  ASSERT_EQ(matching_cost->GetHeight(), aggregate_cost.GetHeight());
  ASSERT_EQ(matching_cost->GetDepth(),  aggregate_cost.GetDepth());

  const int count = matching_cost->GetTotal();
  thrust::device_ptr<const uint16_t> found_ptr(aggregate_cost.GetData());
  thrust::host_vector<uint16_t> found(found_ptr, found_ptr + count);

  for (int i = 0; i < (int)expected.size(); ++i)
  {
    ASSERT_EQ(expected[i], found[i]);
  }
}

TEST(Aggregator, AggregateTopRightToBottomLeft)
{
  std::shared_ptr<MatchingCost> matching_cost = CreateMatchingCost();

  std::vector<uint16_t> expected(matching_cost->GetTotal());
  std::fill(expected.begin(), expected.end(), 0);
  AggregateTopRightToBottomLeft(matching_cost, expected);

  Aggregator aggregator(matching_cost);
  aggregator.SetDirections(Aggregator::DIR_TOP_RIGHT_TO_BOTTOM_LEFT);
  AggregateCost aggregate_cost;
  aggregator.Aggregate(aggregate_cost);

  ASSERT_EQ(matching_cost->GetWidth(),  aggregate_cost.GetWidth());
  ASSERT_EQ(matching_cost->GetHeight(), aggregate_cost.GetHeight());
  ASSERT_EQ(matching_cost->GetDepth(),  aggregate_cost.GetDepth());

  const int count = matching_cost->GetTotal();
  thrust::device_ptr<const uint16_t> found_ptr(aggregate_cost.GetData());
  thrust::host_vector<uint16_t> found(found_ptr, found_ptr + count);

  for (int i = 0; i < (int)expected.size(); ++i)
  {
    ASSERT_EQ(expected[i], found[i]);
  }
}

TEST(Aggregator, AggregateHorizontal)
{
  std::shared_ptr<MatchingCost> matching_cost = CreateMatchingCost();

  std::vector<uint16_t> expected(matching_cost->GetTotal());
  std::fill(expected.begin(), expected.end(), 0);
  AggregateLeftToRight(matching_cost, expected);
  AggregateRightToLeft(matching_cost, expected);

  Aggregator aggregator(matching_cost);
  aggregator.SetDirections(Aggregator::DIR_HORIZONTAL);
  AggregateCost aggregate_cost;
  aggregator.Aggregate(aggregate_cost);

  ASSERT_EQ(matching_cost->GetWidth(),  aggregate_cost.GetWidth());
  ASSERT_EQ(matching_cost->GetHeight(), aggregate_cost.GetHeight());
  ASSERT_EQ(matching_cost->GetDepth(),  aggregate_cost.GetDepth());

  const int count = matching_cost->GetTotal();
  thrust::device_ptr<const uint16_t> found_ptr(aggregate_cost.GetData());
  thrust::host_vector<uint16_t> found(found_ptr, found_ptr + count);

  for (int i = 0; i < (int)expected.size(); ++i)
  {
    ASSERT_EQ(expected[i], found[i]);
  }
}

TEST(Aggregator, AggregateHorizontalVertical)
{
  std::shared_ptr<MatchingCost> matching_cost = CreateMatchingCost();

  std::vector<uint16_t> expected(matching_cost->GetTotal());
  std::fill(expected.begin(), expected.end(), 0);
  AggregateLeftToRight(matching_cost, expected);
  AggregateRightToLeft(matching_cost, expected);
  AggregateTopToBottom(matching_cost, expected);
  AggregateBottomToTop(matching_cost, expected);

  Aggregator aggregator(matching_cost);
  aggregator.SetDirections(Aggregator::DIR_HORIZONTAL_VERTICAL);
  AggregateCost aggregate_cost;
  aggregator.Aggregate(aggregate_cost);

  ASSERT_EQ(matching_cost->GetWidth(),  aggregate_cost.GetWidth());
  ASSERT_EQ(matching_cost->GetHeight(), aggregate_cost.GetHeight());
  ASSERT_EQ(matching_cost->GetDepth(),  aggregate_cost.GetDepth());

  const int count = matching_cost->GetTotal();
  thrust::device_ptr<const uint16_t> found_ptr(aggregate_cost.GetData());
  thrust::host_vector<uint16_t> found(found_ptr, found_ptr + count);

  for (int i = 0; i < (int)expected.size(); ++i)
  {
    ASSERT_EQ(expected[i], found[i]);
  }
}

TEST(Aggregator, AggregateAll)
{
  std::shared_ptr<MatchingCost> matching_cost = CreateMatchingCost();

  std::vector<uint16_t> expected(matching_cost->GetTotal());
  std::fill(expected.begin(), expected.end(), 0);
  AggregateLeftToRight(matching_cost, expected);
  AggregateRightToLeft(matching_cost, expected);
  AggregateTopToBottom(matching_cost, expected);
  AggregateBottomToTop(matching_cost, expected);
  AggregateTopLeftToBottomRight(matching_cost, expected);
  AggregateBottomRightToTopLeft(matching_cost, expected);
  AggregateBottomLeftToTopRight(matching_cost, expected);
  AggregateTopRightToBottomLeft(matching_cost, expected);

  Aggregator aggregator(matching_cost);
  aggregator.SetDirections(Aggregator::DIR_ALL);
  AggregateCost aggregate_cost;
  aggregator.Aggregate(aggregate_cost);

  ASSERT_EQ(matching_cost->GetWidth(),  aggregate_cost.GetWidth());
  ASSERT_EQ(matching_cost->GetHeight(), aggregate_cost.GetHeight());
  ASSERT_EQ(matching_cost->GetDepth(),  aggregate_cost.GetDepth());

  const int count = matching_cost->GetTotal();
  thrust::device_ptr<const uint16_t> found_ptr(aggregate_cost.GetData());
  thrust::host_vector<uint16_t> found(found_ptr, found_ptr + count);

  for (int i = 0; i < (int)expected.size(); ++i)
  {
    ASSERT_EQ(expected[i], found[i]);
  }
}

} // namespace testing

} // namespace matchbox