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

  std::shared_ptr<AggregateCost> cost;
  cost = std::make_shared<AggregateCost>();
  cost->SetSize(w, h, d);

  std::vector<uint16_t> data(cost->GetTotal());
  int index = 0;

  for (int i = 0; i < h; ++i)
  {
    for (int j = 0; j < w; ++j)
    {
      for (int k = 0; k < d; ++k)
      {
        const int pad = ((i + j) % 5 == 0) ? 20 : 0;
        data[index++] = pad + ((i + j + k) % d);
      }
    }
  }

  const cudaMemcpyKind kind = cudaMemcpyHostToDevice;
  const size_t bytes = sizeof(uint16_t) * data.size();
  CUDA_ASSERT(cudaMemcpy(cost->GetData(), data.data(), bytes, kind));
  return cost;
}

inline void Compute(std::shared_ptr<const AggregateCost> cost,
    std::vector<uint8_t>& disparities, float uniqueness = 1.0)
{
  const int w = cost->GetWidth();
  const int h = cost->GetHeight();
  const int d = cost->GetDepth();

  thrust::device_ptr<const uint16_t> ptr(cost->GetData());
  thrust::host_vector<uint16_t> data(ptr, ptr + cost->GetTotal());
  disparities.resize(cost->GetWidth() * cost->GetHeight());

  for (int i = 0; i < h; ++i)
  {
    for (int j = 0; j < w; ++j)
    {
      int min_cost = std::numeric_limits<int>::max();
      int min_disp = 0;

      int sec_min_cost = std::numeric_limits<int>::max();
      int sec_min_disp = 0;

      for (int k = 0; k < d; ++k)
      {
        const int index = (i * w * d) + (j * d) + (k);
        int cost = data[index];

        if (cost < min_cost)
        {
          sec_min_cost = min_cost;
          sec_min_disp = min_disp;
          min_cost = cost;
          min_disp = k;
        }
        else if (cost < sec_min_cost)
        {
          sec_min_cost = cost;
          sec_min_disp = k;
        }
      }

      const bool bad = (uniqueness * sec_min_cost < min_cost) &&
          (abs(sec_min_disp - min_disp) > 1);

      const int index = i * w + j;
      disparities[index] = bad ? 0 : min_disp;
    }
  }
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

  std::vector<uint8_t> expected;
  Compute(cost, expected, 1.0);

  const int w = cost->GetWidth();
  const int h = cost->GetHeight();

  Image image;
  DisparityComputer computer(cost);
  computer.SetUniqueness(1.0);
  computer.SetInverted(false);
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
      ASSERT_EQ((int)expected[index], (int)found[index]);
    }
  }
}

TEST(DisparityComputer, ComputeUnique)
{
  std::shared_ptr<AggregateCost> cost;
  cost = CreateAggregateCost();

  std::vector<uint8_t> expected;
  Compute(cost, expected, 0.95);

  const int w = cost->GetWidth();
  const int h = cost->GetHeight();

  Image image;
  DisparityComputer computer(cost);
  computer.SetUniqueness(0.95);
  computer.SetInverted(false);
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
      ASSERT_EQ((int)expected[index], (int)found[index]);
    }
  }
}

} // namespace testing

} // namespace matchbox