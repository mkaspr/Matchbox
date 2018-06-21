#pragma once

#include <memory>

namespace matchbox
{

class FeatureMap;
class MatchingCost;

class Matcher
{
  public:

    Matcher(std::shared_ptr<const FeatureMap> left_features,
        std::shared_ptr<const FeatureMap> right_features);

    std::shared_ptr<const FeatureMap> GetLeftFeatures() const;

    std::shared_ptr<const FeatureMap> GetRightFeatures() const;

    int GetMaxDisparity() const;

    void SetMaxDisparity(int disparity);

    void Match(MatchingCost& cost) const;

  protected:

    std::shared_ptr<const FeatureMap> left_features_;

    std::shared_ptr<const FeatureMap> right_features_;

    int max_disparity_;
};

} // namespace matchbox