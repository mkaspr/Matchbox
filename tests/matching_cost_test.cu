#include <gtest/gtest.h>
#include <matchbox/exception.h>
#include <matchbox/matching_cost.h>

namespace matchbox
{
namespace testing
{

TEST(MatchingCost, Constructor)
{
  MatchingCost a;
  ASSERT_EQ(0, a.GetTotal());
  ASSERT_EQ(0, a.GetWidth());
  ASSERT_EQ(0, a.GetHeight());
  ASSERT_EQ(0, a.GetDepth());
  ASSERT_EQ(nullptr, a.GetData());

  MatchingCost b(3, 2, 1);
  ASSERT_EQ(6, b.GetTotal());
  ASSERT_EQ(3, b.GetWidth());
  ASSERT_EQ(2, b.GetHeight());
  ASSERT_EQ(1, b.GetDepth());
  ASSERT_NE(nullptr, b.GetData());

  MatchingCost c(b);
  ASSERT_EQ(6, c.GetTotal());
  ASSERT_EQ(3, c.GetWidth());
  ASSERT_EQ(2, c.GetHeight());
  ASSERT_EQ(1, c.GetDepth());
  ASSERT_NE(b.GetData(), c.GetData());
}

TEST(MatchingCost, Size)
{
  MatchingCost cost;
  uint8_t* data;

  cost.SetSize(3, 2, 1);
  data = cost.GetData();
  ASSERT_EQ(6, cost.GetTotal());
  ASSERT_EQ(3, cost.GetWidth());
  ASSERT_EQ(2, cost.GetHeight());
  ASSERT_EQ(1, cost.GetDepth());
  ASSERT_NE(nullptr, cost.GetData());

  cost.SetSize(2, 1, 3);
  ASSERT_EQ(6, cost.GetTotal());
  ASSERT_EQ(2, cost.GetWidth());
  ASSERT_EQ(1, cost.GetHeight());
  ASSERT_EQ(3, cost.GetDepth());
  ASSERT_EQ(data, cost.GetData());

  cost.SetSize(0, 1, 3);
  ASSERT_EQ(0, cost.GetTotal());
  ASSERT_EQ(0, cost.GetWidth());
  ASSERT_EQ(1, cost.GetHeight());
  ASSERT_EQ(3, cost.GetDepth());
  ASSERT_EQ(nullptr, cost.GetData());

#ifndef NDEBUG
  ASSERT_THROW(cost.SetSize(-1,  1,  1), Exception);
  ASSERT_THROW(cost.SetSize( 1, -1,  1), Exception);
  ASSERT_THROW(cost.SetSize( 1,  1, -1), Exception);
#endif
}

} // namespace testing

} // namespace matchbox