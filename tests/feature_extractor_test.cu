#include <memory>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <matchbox/device.h>
#include <matchbox/feature_extractor.h>
#include <matchbox/feature_map.h>
#include <matchbox/image.h>

#include <bitset>

namespace matchbox
{
namespace testing
{
namespace
{

inline cv::Mat CreateMat()
{
  cv::Mat mat(480, 640, CV_8UC1);

  for (int y = 0; y < mat.rows; ++y)
  {
    for (int x = 0; x < mat.cols; ++x)
    {
      mat.at<uint8_t>(y, x) = 25 * (y % 9) + x % 23 + x % 3;
    }
  }

  return mat;
}

inline uint64_t Extract(const cv::Mat& mat, int x, int y)
{
  uint64_t feature = 0;
  const uint8_t center = mat.at<uint8_t>(y, x);

  for (int j = -3; j <= 3; ++j)
  {
    for (int i = -4; i <= 4; ++i)
    {
      uint8_t other = 0;

      if (y + j >= 0 && y + j < mat.rows &&
          x + i >= 0 && x + i < mat.cols)
      {
        other = mat.at<uint8_t>(y + j, x + i);
      }

      feature <<= 1;
      feature |= center >= other;
    }
  }

  return feature;
}

inline cv::Mat Extract(const cv::Mat& mat)
{
  cv::Mat expected(mat.rows, mat.cols, CV_64FC1);

  for (int y = 0; y < expected.rows; ++y)
  {
    for (int x = 0; x < expected.cols; ++x)
    {
      expected.at<uint64_t>(y, x) = Extract(mat, x, y);
    }
  }

  return expected;
}

} // namespace

TEST(FeatureExtractor, Constructor)
{
  std::shared_ptr<Image> image;
  image = std::make_shared<Image>();
  FeatureExtractor extractor(image);
  ASSERT_EQ(image, extractor.GetImage());
}

TEST(FeatureExtractor, Extract)
{
  const cv::Mat mat = CreateMat();
  const cv::Mat expected = Extract(mat);

  std::shared_ptr<Image> image;
  image = std::make_shared<Image>(mat.cols, mat.rows);
  const cudaMemcpyKind kind = cudaMemcpyHostToDevice;
  CUDA_ASSERT(cudaMemcpy(image->GetData(), mat.data, image->GetTotal(), kind));

  FeatureMap map;
  FeatureExtractor extractor(image);
  extractor.Extract(map);

  thrust::device_ptr<uint64_t> ptr(map.GetData());
  thrust::host_vector<uint64_t> found(ptr, ptr + map.GetTotal());
  const uint64_t* data = reinterpret_cast<const uint64_t*>(expected.data);

  for (int i = 0; i < (int)found.size(); ++i)
  {
    if (data[i] != found[i])
    {
      const int x = i % image->GetWidth();
      const int y = i / image->GetWidth();
      std::cout << i << " (" << x << ", " << y<< "): " << data[i] << " " << found[i] << std::endl;
    }

    ASSERT_EQ(data[i], found[i]);
  }
}

} // namespace testing

} // namespace matchbox