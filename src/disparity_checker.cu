#include <matchbox/disparity_checker.h>
#include <matchbox/device.h>
#include <matchbox/exception.h>
#include <matchbox/image.h>

namespace matchbox
{

template <DisparityChecker::Mode mode>
MATCHBOX_DEVICE
bool ShouldCheckLeft()
{
  return mode != DisparityChecker::MODE_CHECK_RIGHT;
}

template <DisparityChecker::Mode mode>
MATCHBOX_DEVICE
bool ShouldCheckRight()
{
  return mode != DisparityChecker::MODE_CHECK_LEFT;
}

MATCHBOX_DEVICE
void CheckLeft(uint8_t* __restrict__ left, uint8_t* __restrict__ right,
    int x, int y, int w, int d)
{
  bool valid = false;
  const uint8_t left_disp = left[y * w + x];

  if (x - left_disp >= 0)
  {
    const uint8_t right_disp = right[y * w + x - left_disp];
    valid = (std::abs(left_disp - right_disp) <= d);
  }

  if (!valid)
  {
    left[y * w + x] = 0;
  }
}

MATCHBOX_DEVICE
void CheckRight(uint8_t* __restrict__ left, uint8_t* __restrict__ right,
    int x, int y, int w, int d)
{
  bool valid = false;
  const uint8_t right_disp = right[y * w + x];

  if (x + right_disp < w)
  {
    const uint8_t left_disp = left[y * w + x + right_disp];
    valid = (std::abs(left_disp - right_disp) <= d);
  }

  if (!valid)
  {
    right[y * w + x] = 0;
  }
}

template <DisparityChecker::Mode mode>
MATCHBOX_GLOBAL
void CheckKernel(uint8_t* __restrict__ left, uint8_t* __restrict__ right,
    int w, int h, int d)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h)
  {
    if (mode != DisparityChecker::MODE_CHECK_RIGHT)
    {
      CheckLeft(left, right, x, y, w, d);
    }

    if (mode != DisparityChecker::MODE_CHECK_LEFT)
    {
      CheckRight(left, right, x, y, w, d);
    }
  }
}

DisparityChecker::DisparityChecker(std::shared_ptr<Image> left_disp,
    std::shared_ptr<Image> right_disp) :
  left_disparities_(left_disp),
  right_disparities_(right_disp),
  max_difference_(1),
  mode_(MODE_CHECK_LEFT)
{
  MATCHBOX_DEBUG(left_disp != right_disp);
  MATCHBOX_DEBUG(left_disp->GetWidth() == right_disp->GetWidth());
  MATCHBOX_DEBUG(left_disp->GetHeight() == right_disp->GetHeight());
}

std::shared_ptr<Image> DisparityChecker::GetLeftDisparities() const
{
  return left_disparities_;
}

std::shared_ptr<Image> DisparityChecker::GetRightDisparities() const
{
  return right_disparities_;
}

int DisparityChecker::GetMaxDifference() const
{
  return max_difference_;
}

void DisparityChecker::SetMaxDifference(int difference)
{
  MATCHBOX_DEBUG(difference >= 0);
  max_difference_ = difference;
}

DisparityChecker::Mode DisparityChecker::GetMode() const
{
  return mode_;
}

void DisparityChecker::SetMode(Mode mode)
{
  mode_ = mode;
}

void DisparityChecker::Check() const
{
  switch (mode_)
  {
    case MODE_CHECK_LEFT:  return Launch<MODE_CHECK_LEFT>();
    case MODE_CHECK_RIGHT: return Launch<MODE_CHECK_RIGHT>();
    case MODE_CHECK_BOTH:  return Launch<MODE_CHECK_BOTH>();
  }
}

template <DisparityChecker::Mode mode>
void DisparityChecker::Launch() const
{
  const int d = max_difference_;
  const int w = left_disparities_->GetWidth();
  const int h = left_disparities_->GetHeight();
  uint8_t* left = left_disparities_->GetData();
  uint8_t* right = right_disparities_->GetData();

  const dim3 threads(w, h);
  const dim3 blocks(32, 32);
  const dim3 grids = GetGrids(threads, blocks);

  CUDA_LAUNCH(CheckKernel<mode>, grids, blocks, 0, 0, left, right, w, h, d);
}

} // namespace matchbox