#include <matchbox/median_filter.h>
#include <matchbox/device.h>
#include <matchbox/exception.h>
#include <matchbox/image.h>

#define BLOCK_DIM 32
// #define SHARED_SIZE 1156

namespace matchbox
{

MATCHBOX_GLOBAL
void FilterKernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int w, int h)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h)
  {
    int count = 0;
    uint8_t kernel[9];

    for (int i = -1; i <= 1; ++i)
    {
      if (y + i < 0 || y + i >= h) continue;

      for (int j = -1; j <= 1; ++j)
      {
        if (x + j < 0 || x + j >= w) continue;
        const int index = (y + i) * w + (x + j);
        kernel[count++] = src[index];
      }
    }

    int mindex = count / 2;

    for (int i = 0; i <= mindex; ++i)
    {
      for (int j = i + 1; j < count; ++j)
      {
        if (kernel[i] > kernel[j])
        {
          uint8_t temp = kernel[i];
          kernel[i] = kernel[j];
          kernel[j] = temp;
        }
      }
    }

    const int index = y * w + x;
    dst[index] = kernel[mindex];

    // const int index = y * w + x;
    // dst[index] = src[index];
  }
}

MedianFilter::MedianFilter(std::shared_ptr<const Image> source) :
  source_(source)
{
}

std::shared_ptr<const Image> MedianFilter::GetSourceImage() const
{
  return source_;
}

void MedianFilter::Filter(Image& output) const
{
  const int w = source_->GetWidth();
  const int h = source_->GetHeight();

  output.SetSize(w, h);
  uint8_t* dst = output.GetData();
  const uint8_t* src = source_->GetData();

  const dim3 threads(w, h);
  const dim3 blocks(BLOCK_DIM, BLOCK_DIM);
  const dim3 grids = GetGrids(threads, blocks);

  CUDA_LAUNCH(FilterKernel, grids, blocks, 0, 0, src, dst, w, h);
}

} // namespace matchbox