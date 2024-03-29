#include <gtest/gtest.h>
#include <matchbox/aggregate_cost.h>
#include <matchbox/exception.h>

namespace matchbox
{
namespace testing
{

TEST(AggregateCost, Constructor)
{
  AggregateCost a;
  ASSERT_EQ(0, a.GetBytes());
  ASSERT_EQ(0, a.GetTotal());
  ASSERT_EQ(0, a.GetWidth());
  ASSERT_EQ(0, a.GetHeight());
  ASSERT_EQ(0, a.GetDepth());
  ASSERT_EQ(nullptr, a.GetData());

  AggregateCost b(3, 4, 6);
  ASSERT_EQ(144, b.GetBytes());
  ASSERT_EQ(72, b.GetTotal());
  ASSERT_EQ(3, b.GetWidth());
  ASSERT_EQ(4, b.GetHeight());
  ASSERT_EQ(6, b.GetDepth());
  ASSERT_NE(nullptr, b.GetData());

  AggregateCost c(b);
  ASSERT_EQ(144, c.GetBytes());
  ASSERT_EQ(72, c.GetTotal());
  ASSERT_EQ(3, c.GetWidth());
  ASSERT_EQ(4, c.GetHeight());
  ASSERT_EQ(6, c.GetDepth());
  ASSERT_NE(b.GetData(), c.GetData());
}

TEST(AggregateCost, Size)
{
  AggregateCost cost;
  uint16_t* data;

  cost.SetSize(3, 4, 6);
  data = cost.GetData();
  ASSERT_EQ(144, cost.GetBytes());
  ASSERT_EQ(72, cost.GetTotal());
  ASSERT_EQ(3, cost.GetWidth());
  ASSERT_EQ(4, cost.GetHeight());
  ASSERT_EQ(6, cost.GetDepth());
  ASSERT_NE(nullptr, cost.GetData());

  cost.SetSize(3, 2, 4);
  ASSERT_EQ(48, cost.GetBytes());
  ASSERT_EQ(24, cost.GetTotal());
  ASSERT_EQ(3, cost.GetWidth());
  ASSERT_EQ(2, cost.GetHeight());
  ASSERT_EQ(4, cost.GetDepth());
  ASSERT_EQ(data, cost.GetData());

#ifndef NDEBUG
  ASSERT_THROW(cost.SetSize(-1,  1,  1), Exception);
  ASSERT_THROW(cost.SetSize( 1, -1,  1), Exception);
  ASSERT_THROW(cost.SetSize( 1,  1, -1), Exception);
#endif
}

} // namespace testing

} // namespace matchbox