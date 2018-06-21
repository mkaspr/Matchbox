#include <gtest/gtest.h>
#include <matchbox/exception.h>
#include <matchbox/feature_map.h>

namespace matchbox
{
namespace testing
{

TEST(FeatureMap, Constructor)
{
  FeatureMap a;
  ASSERT_EQ(0, a.GetBytes());
  ASSERT_EQ(0, a.GetTotal());
  ASSERT_EQ(0, a.GetWidth());
  ASSERT_EQ(0, a.GetHeight());
  ASSERT_EQ(nullptr, a.GetData());

  FeatureMap b(3, 2);
  ASSERT_EQ(48, b.GetBytes());
  ASSERT_EQ(6, b.GetTotal());
  ASSERT_EQ(3, b.GetWidth());
  ASSERT_EQ(2, b.GetHeight());
  ASSERT_NE(nullptr, b.GetData());

  FeatureMap c(b);
  ASSERT_EQ(48, c.GetBytes());
  ASSERT_EQ(6, c.GetTotal());
  ASSERT_EQ(3, c.GetWidth());
  ASSERT_EQ(2, c.GetHeight());
  ASSERT_NE(b.GetData(), c.GetData());
}

TEST(FeatureMap, Size)
{
  FeatureMap map;
  uint64_t* data;

  map.SetSize(3, 2);
  data = map.GetData();
  ASSERT_EQ(48, map.GetBytes());
  ASSERT_EQ(6, map.GetTotal());
  ASSERT_EQ(3, map.GetWidth());
  ASSERT_EQ(2, map.GetHeight());
  ASSERT_NE(nullptr, map.GetData());

  map.SetSize(2, 3);
  ASSERT_EQ(48, map.GetBytes());
  ASSERT_EQ(6, map.GetTotal());
  ASSERT_EQ(2, map.GetWidth());
  ASSERT_EQ(3, map.GetHeight());
  ASSERT_EQ(data, map.GetData());

  map.SetSize(0, 0);
  ASSERT_EQ(0, map.GetBytes());
  ASSERT_EQ(0, map.GetTotal());
  ASSERT_EQ(0, map.GetWidth());
  ASSERT_EQ(0, map.GetHeight());
  ASSERT_EQ(nullptr, map.GetData());

#ifndef NDEBUG
  ASSERT_THROW(map.SetSize(-1,  1), Exception);
  ASSERT_THROW(map.SetSize( 1, -1), Exception);
  ASSERT_THROW(map.SetSize(-1, -1), Exception);
#endif
}

} // namespace testing

} // namespace matchbox