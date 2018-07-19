#pragma once

#include <memory>

namespace matchbox
{

class DepthImage;
class Image;

struct Calibration
{
  float baseline;

  float focal_length;

  float left_center_point;

  float right_center_point;
};

class DisparityConverter
{
  public:

    DisparityConverter(std::shared_ptr<const Image> disparities);

    std::shared_ptr<const Image> GetDisparities() const;

    const Calibration& GetCalibration() const;

    void SetCalibration(const Calibration& calibration);

    void Convert(DepthImage& depth) const;

  protected:

    Calibration calibration_;

    std::shared_ptr<const Image> disparities_;
};

} // namespace matchbox