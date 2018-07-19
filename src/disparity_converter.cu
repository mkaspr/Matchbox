#include <matchbox/disparity_converter.h>
#include <matchbox/device.h>
#include <matchbox/image.h>

namespace matchbox
{

MATCHBOX_GLOBAL
void ConvertKernel(const uint8_t* disparities, float* depths, int w, int h,
  const Calibration calib)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h)
  {
    const int index = y * w + x;
    const uint8_t disparity = disparities[index];
    float depth = 0.0f;

    if (disparity > 0 && disparity < x)
    {
      const float b = calib.baseline;
      const float f = calib.focal_length;
      const float dl = 0.5f + x - calib.left_center_point;
      const float dr = 0.5f + x - disparity - calib.right_center_point;
      depth = (b * f) / (dl - dr);
    }

    depths[index] = depth;
  }
}

DisparityConverter::DisparityConverter(
    std::shared_ptr<const Image> disparities) :
  disparities_(disparities)
{
  calibration_.baseline = 0.10;
  calibration_.focal_length = 500;
  calibration_.left_center_point = 320;
  calibration_.right_center_point = 320;
}

std::shared_ptr<const Image> DisparityConverter::GetDisparities() const
{
  return disparities_;
}

const Calibration& DisparityConverter::GetCalibration() const
{
  return calibration_;
}

void DisparityConverter::SetCalibration(const Calibration& calibration)
{
  calibration_ = calibration;
}

void DisparityConverter::Convert(DepthImage& depth) const
{
  const int w = disparities_->GetWidth();
  const int h = disparities_->GetHeight();

  depth.SetSize(w, h);
  float* depths = depth.GetData();
  const uint8_t* disparities = disparities_->GetData();

  const dim3 total(w, h);
  const dim3 threads(16, 16);
  const dim3 blocks = GetGrids(total, threads);

  CUDA_LAUNCH(ConvertKernel, blocks, threads, 0, 0, disparities, depths, w, h,
      calibration_);
}

} // namespace matchbox