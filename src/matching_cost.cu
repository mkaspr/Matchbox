#include <matchbox/matching_cost.h>
#include <matchbox/device.h>
#include <matchbox/exception.h>

namespace matchbox
{

MatchingCost::MatchingCost() :
  width_(0),
  height_(0),
  depth_(0),
  data_(nullptr)
{
}

MatchingCost::MatchingCost(int w, int h, int d) :
  width_(0),
  height_(0),
  depth_(0),
  data_(nullptr)
{
  SetSize(w, h, d);
}

MatchingCost::MatchingCost(const MatchingCost& cost) :
  width_(0),
  height_(0),
  depth_(0),
  data_(nullptr)
{
  *this = cost;
}

MatchingCost& MatchingCost::operator=(const MatchingCost& cost)
{
  SetSize(cost.GetWidth(), cost.GetHeight(), cost.GetDepth());
  const cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
  CUDA_DEBUG(cudaMemcpy(data_, cost.GetData(), GetTotal(), kind));
  return *this;
}

MatchingCost::~MatchingCost()
{
  cudaFree(data_);
}

int MatchingCost::GetTotal() const
{
  return width_ * height_ * depth_;
}

int MatchingCost::GetWidth() const
{
  return width_;
}

int MatchingCost::GetHeight() const
{
  return height_;
}

int MatchingCost::GetDepth() const
{
  return depth_;
}

void MatchingCost::SetSize(int w, int h, int d)
{
  MATCHBOX_DEBUG(w >= 0 && h >= 0 && d >= 0);
  const int curr_total = GetTotal();
  width_ = w; height_ = h; depth_ = d;
  const int new_total = GetTotal();

  if (new_total != curr_total)
  {
    CUDA_DEBUG(cudaFree(data_));
    CUDA_DEBUG(cudaMalloc(&data_, new_total));
  }
}

const uint8_t* MatchingCost::GetData() const
{
  return data_;
}

uint8_t* MatchingCost::GetData()
{
  return data_;
}

} // namespace matchbox