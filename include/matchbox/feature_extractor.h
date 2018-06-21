#pragma once

#include <memory>

namespace matchbox
{

class FeatureMap;
class Image;

class FeatureExtractor
{
  public:

    FeatureExtractor(std::shared_ptr<const Image> image);

    std::shared_ptr<const Image> GetImage() const;

    void Extract(FeatureMap& map) const;

  protected:

    std::shared_ptr<const Image> image_;
};

} // namespace matchbox