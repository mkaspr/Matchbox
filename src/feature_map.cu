#include <matchbox/feature_map.h>
#include <matchbox/device.h>
#include <matchbox/exception.h>

namespace matchbox
{

FeatureMap::FeatureMap() :
  width_(0),
  height_(0),
  data_(nullptr)
{
}

FeatureMap::FeatureMap(int w, int h) :
  width_(0),
  height_(0),
  data_(nullptr)
{
  SetSize(w, h);
}

FeatureMap::FeatureMap(const FeatureMap& map) :
  width_(0),
  height_(0),
  data_(nullptr)
{
  *this = map;
}

FeatureMap& FeatureMap::operator=(const FeatureMap& map)
{
  SetSize(map.GetWidth(), map.GetHeight());
  const cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
  CUDA_DEBUG(cudaMemcpy(data_, map.GetData(), GetTotal(), kind));
  return *this;
}

FeatureMap::~FeatureMap()
{
  cudaFree(data_);
}

size_t FeatureMap::GetBytes() const
{
  return sizeof(uint64_t) * GetTotal();
}

int FeatureMap::GetTotal() const
{
  return width_ * height_;
}

int FeatureMap::GetWidth() const
{
  return width_;
}

int FeatureMap::GetHeight() const
{
  return height_;
}

void FeatureMap::SetSize(int w, int h)
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

const uint64_t* FeatureMap::GetData() const
{
  return data_;
}

uint64_t* FeatureMap::GetData()
{
  return data_;
}

} // namespace matchbox