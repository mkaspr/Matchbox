#include <matchbox/image.h>
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

void Image::Load(const cv::Mat& image)
{
  cv::Mat src = image.clone();

  if (image.channels() == 3)
  {
    cv::cvtColor(src, src, CV_RGB2GRAY);
  }

  MATCHBOX_ASSERT_MSG(src.data, "unable to load source");
  MATCHBOX_ASSERT_MSG(src.elemSize() == 1, "invalid source");
  SetSize(src.cols, src.rows);
  CUDA_DEBUG(cudaMemcpy(data_, src.data, GetTotal(), cudaMemcpyHostToDevice));
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

////////////////////////////////////////////////////////////////////////////////

DepthImage::DepthImage() :
  width_(0),
  height_(0),
  data_(nullptr)
{
}

DepthImage::DepthImage(int w, int h) :
  width_(0),
  height_(0),
  data_(nullptr)
{
  SetSize(w, h);
}

DepthImage::DepthImage(const DepthImage& image) :
  width_(0),
  height_(0),
  data_(nullptr)
{
  *this = image;
}

DepthImage& DepthImage::operator=(const DepthImage& image)
{
  SetSize(image.GetWidth(), image.GetHeight());
  const cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
  CUDA_DEBUG(cudaMemcpy(data_, image.GetData(), GetBytes(), kind));
  return *this;
}

DepthImage::~DepthImage()
{
  cudaFree(data_);
}

int DepthImage::GetBytes() const
{
  return sizeof(float) * GetTotal();
}

int DepthImage::GetTotal() const
{
  return width_ * height_;
}

int DepthImage::GetWidth() const
{
  return width_;
}

int DepthImage::GetHeight() const
{
  return height_;
}

void DepthImage::SetSize(int w, int h)
{
  MATCHBOX_DEBUG(w >= 0 && h >= 0);
  const int curr_total = GetTotal();
  width_ = w; height_ = h;
  const int new_total = GetTotal();

  if (new_total != curr_total)
  {
    CUDA_DEBUG(cudaFree(data_));
    CUDA_DEBUG(cudaMalloc(&data_, GetBytes()));
  }
}

void DepthImage::Save(const std::string& file) const
{
  cv::Mat image(height_, width_, CV_32FC1);
  CUDA_DEBUG(cudaMemcpy(image.data, data_, GetBytes(), cudaMemcpyDeviceToHost));
  image.convertTo(image, CV_16UC1, 1000);
  cv::imwrite(file, image);
}

const float* DepthImage::GetData() const
{
  return data_;
}

float* DepthImage::GetData()
{
  return data_;
}

} // namespace matchbox