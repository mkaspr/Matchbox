#include <bitset>
#include <gtest/gtest.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <matchbox/device.h>
#include <matchbox/exception.h>
#include <matchbox/feature_map.h>
#include <matchbox/matching_cost.h>
#include <matchbox/matcher.h>

#include <ctime>

namespace matchbox
{
namespace testing
{
namespace
{

inline std::shared_ptr<FeatureMap> CreateLeftFeatures()
{
  std::shared_ptr<FeatureMap> features;
  features = std::make_shared<FeatureMap>(640, 480);
  const int w = features->GetWidth();
  const int h = features->GetHeight();
  std::vector<uint64_t> data(w * h);
  int index = 0;

  for (int y = 0; y < h; ++y)
  {
    for (int x = 0; x < w; ++x)
    {
      data[index++] = y + x;
    }
  }

  const size_t bytes = features->GetBytes();
  const cudaMemcpyKind kind = cudaMemcpyHostToDevice;
  CUDA_ASSERT(cudaMemcpy(features->GetData(), data.data(), bytes, kind));
  return features;
}

inline std::shared_ptr<FeatureMap> CreateRightFeatures()
{
  std::shared_ptr<FeatureMap> features;
  features = std::make_shared<FeatureMap>(640, 480);
  const int w = features->GetWidth();
  const int h = features->GetHeight();
  std::vector<uint64_t> data(w * h);
  int index = 0;

  for (int y = 0; y < h; ++y)
  {
    for (int x = 0; x < w; ++x)
    {
      data[index++] = 2 * y + x;
    }
  }

  const size_t bytes = features->GetBytes();
  const cudaMemcpyKind kind = cudaMemcpyHostToDevice;
  CUDA_ASSERT(cudaMemcpy(features->GetData(), data.data(), bytes, kind));
  return features;
}

inline std::shared_ptr<MatchingCost> Match(int disp,
  std::shared_ptr<const FeatureMap> left,
  std::shared_ptr<const FeatureMap> right)
{
  const int w = left->GetWidth();
  const int h = right->GetHeight();
  std::shared_ptr<MatchingCost> cost;
  cost = std::make_shared<MatchingCost>(w, h, disp);

  std::vector<uint8_t> data(w * h * disp);
  thrust::device_ptr<const uint64_t> left_ptr(left->GetData());
  thrust::device_ptr<const uint64_t> right_ptr(right->GetData());
  thrust::host_vector<uint64_t> left_data(left_ptr, left_ptr + w * h);
  thrust::host_vector<uint64_t> right_data(right_ptr, right_ptr + w * h);

  int index = 0;

  for (int y = 0; y < h; ++y)
  {
    for (int x = 0; x < w; ++x)
    {
      const uint64_t l = left_data[y * w + x];

      for (int d = 0; d < disp; ++d)
      {
        if (x - d < 0)
        {
          data[index++] = 64;
        }
        else
        {
          // const uint64_t r = (x - d >= 0) ? right_data[y * w + x - d] : 0;
          const uint64_t r = right_data[y * w + x - d];
          data[index++] = std::bitset<64>(l ^ r).count();
        }
      }
    }
  }

  const cudaMemcpyKind kind = cudaMemcpyHostToDevice;
  CUDA_ASSERT(cudaMemcpy(cost->GetData(), data.data(), cost->GetTotal(), kind));
  return cost;
}

} // namespace

TEST(Matcher, Constructor)
{
  std::shared_ptr<FeatureMap> left_features;
  std::shared_ptr<FeatureMap> right_features;
  left_features = std::make_shared<FeatureMap>(9, 7);
  right_features = std::make_shared<FeatureMap>(9, 7);
  Matcher matcher(left_features, right_features);
  ASSERT_EQ(left_features, matcher.GetLeftFeatures());
  ASSERT_EQ(right_features, matcher.GetRightFeatures());
  ASSERT_EQ(128, matcher.GetMaxDisparity());
}

TEST(Matcher, Match)
{
  std::shared_ptr<FeatureMap> left = CreateLeftFeatures();
  std::shared_ptr<FeatureMap> right = CreateRightFeatures();

  Matcher matcher(left, right);
  const int disp = matcher.GetMaxDisparity();
  std::shared_ptr<MatchingCost> expected_cost = Match(disp, left, right);

  MatchingCost found_cost;
  matcher.Match(found_cost);

  const int count = expected_cost->GetTotal();
  ASSERT_EQ(expected_cost->GetTotal(), found_cost.GetTotal());
  thrust::device_ptr<const uint8_t> expected_ptr(expected_cost->GetData());
  thrust::device_ptr<const uint8_t> found_ptr(found_cost.GetData());
  thrust::host_vector<uint8_t> expected(expected_ptr, expected_ptr + count);
  thrust::host_vector<uint8_t> found(found_ptr, found_ptr + count);

  for (size_t i = 0 ; i < expected.size(); ++i)
  {
    if (expected[i] != found[i])
    {
      std::cout << i << ": " << (int)expected[i] << " " << (int)found[i] << std::endl;
    }

    ASSERT_EQ(expected[i], found[i]);
  }
}

} // namespace testing

} // namespace matchbox