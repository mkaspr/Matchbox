#include <matchbox/median_filter.h>
#include <matchbox/device.h>
#include <matchbox/exception.h>
#include <matchbox/image.h>

#define BLOCK_DIM 16

namespace matchbox
{

MATCHBOX_GLOBAL
void FilterKernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int w, int h)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int index = y * w + x;

  if (x > 1 && y > 1 && x < w - 1 && y < h - 1)
  {
    uint8_t kernel[9];

    for (int i = -1; i <= 1; ++i)
    {
      for (int j = -1; j <= 1; ++j)
      {
        const int index = (y + i) * w + (x + j);
        kernel[(i+1)*3+(j+1)] = src[index];
      }
    }

    int mindex = 4;

    for (int i = 0; i <= mindex; ++i)
    {
      int min_index = 0;

      for (int j = i + 1; j < 9; ++j)
      {
        if (kernel[j] < kernel[min_index])
        {
          min_index = j;
        }
      }

      const uint8_t temp = kernel[i];
      kernel[i] = kernel[min_index];
      kernel[min_index] = temp;
    }

    dst[index] = kernel[mindex];
  }
  else if (x < w && y < h)
  {
    dst[index] = src[index];
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