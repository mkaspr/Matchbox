#pragma once

#include <memory>

namespace matchbox
{

class Image;

class MedianFilter
{
  public:

    MedianFilter(std::shared_ptr<const Image> source);

    std::shared_ptr<const Image> GetSourceImage() const;

    void Filter(Image& output) const;

  protected:

    std::shared_ptr<const Image> source_;
};

} // namespace matchbox