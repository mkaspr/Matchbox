#pragma once

#include <memory>

namespace matchbox
{

class Image;

class DisparityChecker
{
  public:

    enum Mode
    {
      MODE_CHECK_LEFT,
      MODE_CHECK_RIGHT,
      MODE_CHECK_BOTH,
    };

  public:

    DisparityChecker(std::shared_ptr<Image> left_disp,
        std::shared_ptr<Image> right_disp);

    std::shared_ptr<Image> GetLeftDisparities() const;

    std::shared_ptr<Image> GetRightDisparities() const;

    int GetMaxDifference() const;

    void SetMaxDifference(int difference);

    Mode GetMode() const;

    void SetMode(Mode mode);

    void Check() const;

  protected:

    template <Mode mode>
    void Launch() const;

  protected:

    std::shared_ptr<Image> left_disparities_;

    std::shared_ptr<Image> right_disparities_;

    int max_difference_;

    Mode mode_;
};

} // namespace matchbox