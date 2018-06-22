#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <thrust/device_ptr.h>
#include <matchbox/device.h>
#include <matchbox/image.h>

namespace matchbox
{
namespace testing
{

TEST(Image, Constructor)
{
  Image a;
  ASSERT_EQ(0, a.GetTotal());
  ASSERT_EQ(0, a.GetWidth());
  ASSERT_EQ(0, a.GetHeight());
  ASSERT_EQ(nullptr, a.GetData());

  Image b(3, 2);
  ASSERT_EQ(6, b.GetTotal());
  ASSERT_EQ(3, b.GetWidth());
  ASSERT_EQ(2, b.GetHeight());
  ASSERT_NE(nullptr, b.GetData());

  Image c(b);
  ASSERT_EQ(6, c.GetTotal());
  ASSERT_EQ(3, c.GetWidth());
  ASSERT_EQ(2, c.GetHeight());
  ASSERT_NE(b.GetData(), c.GetData());
}

TEST(Image, Size)
{
  Image image;
  uint8_t* data;

  image.SetSize(1, 1);
  ASSERT_EQ(1, image.GetTotal());
  ASSERT_EQ(1, image.GetWidth());
  ASSERT_EQ(1, image.GetHeight());
  ASSERT_NE(nullptr, image.GetData());

  image.SetSize(3, 2);
  data = image.GetData();
  ASSERT_EQ(6, image.GetTotal());
  ASSERT_EQ(3, image.GetWidth());
  ASSERT_EQ(2, image.GetHeight());
  ASSERT_NE(nullptr, image.GetData());

  image.SetSize(2, 3);
  ASSERT_EQ(6, image.GetTotal());
  ASSERT_EQ(2, image.GetWidth());
  ASSERT_EQ(3, image.GetHeight());
  ASSERT_EQ(data, image.GetData());

  image.SetSize(0, 0);
  ASSERT_EQ(0, image.GetTotal());
  ASSERT_EQ(0, image.GetWidth());
  ASSERT_EQ(0, image.GetHeight());
  ASSERT_EQ(nullptr, image.GetData());

#ifndef NDEBUG
  ASSERT_THROW(image.SetSize(-1,  1), Exception);
  ASSERT_THROW(image.SetSize( 1, -1), Exception);
  ASSERT_THROW(image.SetSize(-1, -1), Exception);
#endif
}

TEST(Image, Load)
{
  Image image;
  cv::Mat expected(7, 9, CV_8UC1);

  for (int i = 0; i < (int)expected.total(); ++i)
  {
    expected.data[i] = i;
  }

  const std::string file("temp.png");
  cv::imwrite(file, expected);

  image.Load(file);
  std::remove(file.c_str());

  ASSERT_EQ(9, image.GetWidth());
  ASSERT_EQ(7, image.GetHeight());
  ASSERT_EQ(63, image.GetTotal());
  thrust::device_ptr<uint8_t> ptr(image.GetData());

  for (int i = 0; i < image.GetTotal(); ++i)
  {
    ASSERT_EQ(expected.data[i], ptr[i]);
  }
}

TEST(Image, Save)
{
  Image image(9, 7);
  thrust::device_ptr<uint8_t> ptr(image.GetData());

  for (int i = 0; i < image.GetTotal(); ++i)
  {
    ptr[i] = i;
  }

  const std::string file("temp.png");
  image.Save(file);
  cv::Mat found = cv::imread(file, CV_LOAD_IMAGE_GRAYSCALE);
  std::remove(file.c_str());

  ASSERT_EQ(9, found.cols);
  ASSERT_EQ(7, found.rows);

  for (int i = 0; i < (int)found.total(); ++i)
  {
    ASSERT_EQ(i, found.data[i]);
  }
}

} // namespace testing

} // namespace matchbox