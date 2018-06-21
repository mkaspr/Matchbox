#include <gtest/gtest.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <matchbox/aggregate_cost.h>
#include <matchbox/device.h>
#include <matchbox/disparity_computer.h>
#include <matchbox/exception.h>
#include <matchbox/image.h>

namespace matchbox
{
namespace testing
{
namespace
{

inline std::shared_ptr<AggregateCost> CreateAggregateCost()
{
  const int w = 64;
  const int h = 32;
  const int d = 128;
  const int p = 2;

  std::shared_ptr<AggregateCost> cost;
  cost = std::make_shared<AggregateCost>();
  cost->SetSize(w, h, d, p);

  std::vector<uint16_t> data(cost->GetTotal());
  int index = 0;

  for (int i = 0; i < h; ++i)
  {
    for (int j = 0; j < w; ++j)
    {
      for (int k = 0; k < d; ++k)
      {
        data[index++] = (1 + i + j + k) % d;
      }
    }
  }

  for (int i = 0; i < h; ++i)
  {
    for (int j = 0; j < w; ++j)
    {
      for (int k = 0; k < d; ++k)
      {
        data[index++] = 2 * (i + j + k) % d;
      }
    }
  }

  const cudaMemcpyKind kind = cudaMemcpyHostToDevice;
  const size_t bytes = sizeof(uint16_t) * data.size();
  CUDA_ASSERT(cudaMemcpy(cost->GetData(), data.data(), bytes, kind));
  return cost;
}

} // namespace

TEST(DisparityComputer, Constructor)
{
  std::shared_ptr<AggregateCost> cost;
  cost = std::make_shared<AggregateCost>();
  DisparityComputer computer(cost);
  ASSERT_EQ(cost, computer.GetCost());
  ASSERT_FLOAT_EQ(0.95, computer.GetUniqueness());
}

TEST(DisparityComputer, Uniquess)
{
  std::shared_ptr<AggregateCost> cost;
  cost = std::make_shared<AggregateCost>();
  DisparityComputer computer(cost);

  computer.SetUniqueness(0.80);
  ASSERT_FLOAT_EQ(0.80, computer.GetUniqueness());

  computer.SetUniqueness(1.00);
  ASSERT_FLOAT_EQ(1.00, computer.GetUniqueness());

#ifndef NDEBUG
  ASSERT_THROW(computer.SetUniqueness(-0.50), Exception);
  ASSERT_THROW(computer.SetUniqueness( 1.50), Exception);
#endif
}

TEST(DisparityComputer, Compute)
{
  std::shared_ptr<AggregateCost> cost;
  cost = CreateAggregateCost();

  const int w = cost->GetWidth();
  const int h = cost->GetHeight();
  const int d = cost->GetDepth();

  Image image;
  DisparityComputer computer(cost);
  computer.Compute(image);

  ASSERT_EQ(cost->GetWidth(),  image.GetWidth());
  ASSERT_EQ(cost->GetHeight(), image.GetHeight());

  thrust::device_ptr<const uint8_t> ptr(image.GetData());
  thrust::host_vector<uint8_t> found(ptr, ptr + image.GetTotal());

  for (int i = 0; i < h; ++i)
  {
    for (int j = 0; j < w; ++j)
    {
      const int index = i * w + j;
      const int expected = (128 - i - j) % d;
      ASSERT_EQ((int)expected, (int)found[index]);
    }
  }
}

} // namespace testing

} // namespace matchbox