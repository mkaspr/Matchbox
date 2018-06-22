#include <matchbox/aggregate_cost.h>
#include <matchbox/device.h>
#include <matchbox/exception.h>

namespace matchbox
{

AggregateCost::AggregateCost() :
  width_(0),
  height_(0),
  depth_(0),
  paths_(0),
  data_(nullptr)
{
}

AggregateCost::AggregateCost(int w, int h, int d, int p) :
  width_(0),
  height_(0),
  depth_(0),
  paths_(0),
  data_(nullptr)
{
  SetSize(w, h, d, p);
}

AggregateCost::AggregateCost(const AggregateCost& cost) :
  width_(0),
  height_(0),
  depth_(0),
  paths_(0),
  data_(nullptr)
{
  *this = cost;
}

AggregateCost& AggregateCost::operator=(const AggregateCost& cost)
{
  SetSize(cost.GetWidth(), cost.GetHeight(), cost.GetDepth(), cost.GetPaths());
  const cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
  CUDA_DEBUG(cudaMemcpy(data_, cost.GetData(), GetTotal(), kind));
  return *this;
}

AggregateCost::~AggregateCost()
{
  cudaFree(data_);
}

size_t AggregateCost::GetBytes() const
{
  return sizeof(uint8_t) * GetTotal();
}

int AggregateCost::GetTotal() const
{
  return width_ * height_ * depth_ * paths_;
}

int AggregateCost::GetWidth() const
{
  return width_;
}

int AggregateCost::GetHeight() const
{
  return height_;
}

int AggregateCost::GetDepth() const
{
  return depth_;
}

int AggregateCost::GetPaths() const
{
  return paths_;
}

void AggregateCost::SetSize(int w, int h, int d, int p)
{
  MATCHBOX_DEBUG(w >= 0 && h >= 0 && d >= 0 && p >= 0);
  const int curr_total = GetTotal();
  width_ = w; height_ = h; depth_ = d; paths_ = p;
  const int new_total = GetTotal();

  if (new_total != curr_total)
  {
    CUDA_DEBUG(cudaFree(data_));
    CUDA_DEBUG(cudaMalloc(&data_, GetBytes()));
  }
}

const uint8_t* AggregateCost::GetData() const
{
  return data_;
}

uint8_t* AggregateCost::GetData()
{
  return data_;
}

void AggregateCost::Clear()
{
  CUDA_DEBUG(cudaMemset(data_, 0, GetBytes()));
}

} // namespace matchbox