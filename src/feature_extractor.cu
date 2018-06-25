#include <matchbox/feature_extractor.h>
#include <matchbox/device.h>
#include <matchbox/exception.h>
#include <matchbox/feature_map.h>
#include <matchbox/image.h>

#define BLOCK_SIZE 32
#define SHARED_PAD_X 8
#define SHARED_PAD_Y 6
#define SHARED_DIM_X (BLOCK_SIZE + SHARED_PAD_X)
#define SHARED_DIM_Y (BLOCK_SIZE + SHARED_PAD_Y)
#define SHARED_SIZE (SHARED_DIM_X * SHARED_DIM_Y)

namespace matchbox
{

MATCHBOX_DEVICE
inline void CopyImage(const uint8_t* __restrict__ image, uint8_t* shared,
    int w, int h)
{
  int shared_index = threadIdx.y * blockDim.x + threadIdx.x;
  int x = blockIdx.x * blockDim.x - 4 + shared_index % SHARED_DIM_X;
  int y = blockIdx.y * blockDim.y - 3 + shared_index / SHARED_DIM_X;
  int image_index = y * w + x;

  if (shared_index < SHARED_SIZE)
  {
    shared[shared_index] = (x >= 0 && y >= 0 && x < w && y < h) ? image[image_index] : 0;
    shared_index += BLOCK_SIZE * BLOCK_SIZE;

    if (shared_index < SHARED_SIZE)
    {
      x = blockIdx.x * blockDim.x - 4 + shared_index % SHARED_DIM_X;
      y = blockIdx.y * blockDim.y - 3 + shared_index / SHARED_DIM_X;
      image_index = y * w + x;
      shared[shared_index] = (x >= 0 && y >= 0 && x < w && y < h) ? image[image_index] : 0;
    }
  }
}

MATCHBOX_DEVICE
inline uint64_t ExtractFeature(const uint8_t* shared, int x, int y)
{
  uint64_t feature = 0;
  int index = y * SHARED_DIM_X + x;
  const uint8_t center = shared[index];

  for (int j = -3; j <= 3; ++j)
  {
    for (int i = -4; i <= 4; ++i)
    {
      index = (y + j) * SHARED_DIM_X + (x + i);
      const uint8_t other = shared[index];
      feature <<= 1;
      feature |= center >= other;
    }
  }

  return feature;
}

MATCHBOX_GLOBAL
void ExtractKernel(int w, int h, const uint8_t* __restrict__ image,
  uint64_t* features)
{
  MATCHBOX_SHARED uint8_t shared[SHARED_SIZE];
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  CopyImage(image, shared, w, h);
  __syncthreads();

  if (x < w && y < h)
  {
    const int fx = threadIdx.x + SHARED_PAD_X / 2;
    const int fy = threadIdx.y + SHARED_PAD_Y / 2;
    features[y * w + x] = ExtractFeature(shared, fx, fy);
  }
}

FeatureExtractor::FeatureExtractor(std::shared_ptr<const Image> image) :
  image_(image)
{
}

std::shared_ptr<const Image> FeatureExtractor::GetImage() const
{
  return image_;
}

void FeatureExtractor::Extract(FeatureMap& map) const
{
  const int w = image_->GetWidth();
  const int h = image_->GetHeight();
  const uint8_t* image = image_->GetData();

  const dim3 threads(w, h);
  const dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);
  const dim3 grids = GetGrids(threads, blocks);

  map.SetSize(w, h);
  uint64_t* features = map.GetData();
  CUDA_LAUNCH(ExtractKernel, grids, blocks, 0, 0, w, h, image, features);
}

} // namespace matchbox