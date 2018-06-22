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
  std::vector<uint8_t> data(cost->GetTotal());
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

  return cost;
}

} // namespace

TEST(Aggregator, Constructor)
{
  std::shared_ptr<MatchingCost> matching_cost;
  matching_cost = std::make_shared<MatchingCost>();
  Aggregator aggregator(matching_cost);
  ASSERT_EQ(matching_cost, aggregator.GetMatchingCost());
  ASSERT_EQ(3, aggregator.GetDegree());
}

TEST(Aggregator, AggregateMatching)
{
  std::shared_ptr<MatchingCost> matching_cost = CreateMatchingCost();
  Aggregator aggregator(matching_cost);
  aggregator.SetDegree(0);
  AggregateCost aggregate_cost;
  aggregator.Aggregate(aggregate_cost);

  ASSERT_EQ(matching_cost->GetWidth(),  aggregate_cost.GetWidth());
  ASSERT_EQ(matching_cost->GetHeight(), aggregate_cost.GetHeight());
  ASSERT_EQ(matching_cost->GetDepth(),  aggregate_cost.GetDepth());

  const int count = matching_cost->GetTotal();
  thrust::device_ptr<const uint8_t> expected_ptr(matching_cost->GetData());
  thrust::device_ptr<const uint8_t> found_ptr(aggregate_cost.GetData());
  thrust::host_vector<uint8_t> expected(expected_ptr, expected_ptr + count);
  thrust::host_vector<uint8_t> found(found_ptr, found_ptr + count);

  for (int i = 0; i < (int)expected.size(); ++i)
  {
    ASSERT_EQ(uint8_t(expected[i]), found[i]);
  }
}

TEST(Aggregator, AggregateHorizontal)
{
  std::shared_ptr<MatchingCost> matching_cost = CreateMatchingCost();
  Aggregator aggregator(matching_cost);
  aggregator.SetDegree(1);
  AggregateCost aggregate_cost;
  aggregator.Aggregate(aggregate_cost);

  ASSERT_EQ(matching_cost->GetWidth(),  aggregate_cost.GetWidth());
  ASSERT_EQ(matching_cost->GetHeight(), aggregate_cost.GetHeight());
  ASSERT_EQ(matching_cost->GetDepth(),  aggregate_cost.GetDepth());

  const int count = matching_cost->GetTotal();
  thrust::device_ptr<const uint8_t> expected_ptr(matching_cost->GetData());
  thrust::device_ptr<const uint8_t> found_ptr(aggregate_cost.GetData());
  thrust::host_vector<uint8_t> expected(expected_ptr, expected_ptr + count);
  thrust::host_vector<uint8_t> found(found_ptr, found_ptr + count);

  for (int i = 0; i < (int)expected.size(); ++i)
  {
    ASSERT_EQ(uint8_t(expected[i]), found[i]);
  }
}

} // namespace testing

} // namespace matchbox