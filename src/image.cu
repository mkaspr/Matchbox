#include <matchbox/image.h>
#include <opencv2/opencv.hpp>
#include <matchbox/device.h>

namespace matchbox
{

Image::Image() :
  width_(0),
  height_(0),
  data_(nullptr)
{
}

Image::Image(int w, int h) :
  width_(0),
  height_(0),
  data_(nullptr)
{
  SetSize(w, h);
}

Image::Image(const Image& image) :
  width_(0),
  height_(0),
  data_(nullptr)
{
  *this = image;
}

Image::Image(const std::string& file) :
  width_(0),
  height_(0),
  data_(nullptr)
{
  Load(file);
}

Image& Image::operator=(const Image& image)
{
  SetSize(image.GetWidth(), image.GetHeight());
  const cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
  CUDA_DEBUG(cudaMemcpy(data_, image.GetData(), GetTotal(), kind));
  return *this;
}

Image::~Image()
{
  cudaFree(data_);
}

int Image::GetTotal() const
{
  return width_ * height_;
}

int Image::GetWidth() const
{
  return width_;
}

int Image::GetHeight() const
{
  return height_;
}

void Image::SetSize(int w, int h)
{
  MATCHBOX_DEBUG(w >= 0 && h >= 0);
  const int curr_total = GetTotal();
  width_ = w; height_ = h;
  const int new_total = GetTotal();

  if (new_total != curr_total)
  {
    CUDA_DEBUG(cudaFree(data_));
    CUDA_DEBUG(cudaMalloc(&data_, new_total));
  }
}

void Image::Load(const std::string& file)
{
  cv::Mat image = cv::imread(file, cv::IMREAD_GRAYSCALE);
  MATCHBOX_ASSERT_MSG(image.data, "unable to load image file");
  SetSize(image.cols, image.rows);
  CUDA_DEBUG(cudaMemcpy(data_, image.data, GetTotal(), cudaMemcpyHostToDevice));
}

void Image::Save(const std::string& file) const
{
  cv::Mat image(height_, width_, CV_8UC1);
  CUDA_DEBUG(cudaMemcpy(image.data, data_, GetTotal(), cudaMemcpyDeviceToHost));
  cv::imwrite(file, image);
}

const uint8_t* Image::GetData() const
{
  return data_;
}

uint8_t* Image::GetData()
{
  return data_;
}

} // namespace matchbox