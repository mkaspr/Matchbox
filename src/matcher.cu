#include <matchbox/matcher.h>
#include <matchbox/device.h>
#include <matchbox/exception.h>
#include <matchbox/feature_map.h>
#include <matchbox/matching_cost.h>

#define BLOCK_SIZE 128
#define LEFT_SHARED_SIZE BLOCK_SIZE
#define RIGHT_SHARED_SIZE (2 * BLOCK_SIZE)

namespace matchbox
{

template <int MAX_DISP>
MATCHBOX_DEVICE
inline void CopyLeftFeatures(const uint64_t* __restrict__ src, uint64_t* dst,
    int w, int x, int y)
{
  if (x < w) dst[threadIdx.x] = src[y * w + x];
}

template <int MAX_DISP>
MATCHBOX_DEVICE
inline void CopyRightFeatures(const uint64_t* __restrict__ src, uint64_t* dst,
    int w, int x, int y)
{
  const int x_offset = x - MAX_DISP;
  dst[threadIdx.x] = (x_offset >= 0 && x_offset < w) ? src[y * w + x_offset] : 0;
  dst[MAX_DISP + threadIdx.x] = (x < w) ? src[y * w + x] : 0;
}

template <int MAX_DISP>
MATCHBOX_GLOBAL
void MatchKernel(
    const uint64_t* __restrict__ left_features,
    const uint64_t* __restrict__ right_features,
    uint8_t* cost, int w, int h, int d)
{
  MATCHBOX_SHARED uint64_t shared_left[LEFT_SHARED_SIZE];
  MATCHBOX_SHARED uint64_t shared_right[RIGHT_SHARED_SIZE];

  const int y = blockIdx.y;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int xx = blockIdx.x * blockDim.x;
  CopyLeftFeatures<MAX_DISP>(left_features, shared_left, w, x, y);
  CopyRightFeatures<MAX_DISP>(right_features, shared_right, w, x, y);

  __syncthreads();

  for (int i = 0; i < MAX_DISP; ++i)
  {
    if (xx + i < w)
    {
      // TODO: handle when right disparities are out of index
      // NOTE: don't default to 0 feature, but MAX cost

      if (((xx + i) - int(d - threadIdx.x - 1)) < 0)
      {
        cost[y * w * d + (xx + i) * d + (d - threadIdx.x - 1)] = 64;
      }
      else
      {
        const uint64_t left = shared_left[i];
        const uint64_t right = shared_right[threadIdx.x + 1 + i];
        cost[y * w * d + (xx + i) * d + (d - threadIdx.x - 1)] = __popcll(left ^ right);
      }
    }
  }
}

Matcher::Matcher(std::shared_ptr<const FeatureMap> left_features,
    std::shared_ptr<const FeatureMap> right_features) :
  left_features_(left_features),
  right_features_(right_features),
  max_disparity_(128)
{
  MATCHBOX_DEBUG(left_features->GetWidth() == right_features->GetWidth());
  MATCHBOX_DEBUG(left_features->GetHeight() == right_features->GetHeight());
}

std::shared_ptr<const FeatureMap> Matcher::GetLeftFeatures() const
{
  return left_features_;
}

std::shared_ptr<const FeatureMap> Matcher::GetRightFeatures() const
{
  return right_features_;
}

int Matcher::GetMaxDisparity() const
{
  return max_disparity_;
}

void Matcher::SetMaxDisparity(int disparity)
{
  MATCHBOX_DEBUG(disparity == 128);
  max_disparity_ = disparity;
}

void Matcher::Match(MatchingCost& cost) const
{
  const int w = left_features_->GetWidth();
  const int h = left_features_->GetHeight();

  const dim3 threads(w, h);
  const dim3 blocks(BLOCK_SIZE);
  const dim3 grids = GetGrids(threads, blocks);

  cost.SetSize(w, h, max_disparity_);
  uint8_t* matching_cost = cost.GetData();
  const uint64_t* left_features = left_features_->GetData();
  const uint64_t* right_features = right_features_->GetData();

  CUDA_LAUNCH(MatchKernel<128>, grids, blocks, 0, 0, left_features,
      right_features, matching_cost, w, h, max_disparity_);
}

} // namespace matchbox