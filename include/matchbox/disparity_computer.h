#pragma once

#include <memory>

namespace matchbox
{

class AggregateCost;
class Image;

class DisparityComputer
{
  public:

    DisparityComputer(std::shared_ptr<const AggregateCost> cost);

    std::shared_ptr<const AggregateCost> GetCost() const;

    float GetUniqueness() const;

    void SetUniqueness(float uniqueness);

    bool IsInverted() const;

    void SetInverted(bool inverted);

    void Compute(Image& image) const;

  protected:

    std::shared_ptr<const AggregateCost> cost_;

    float uniqueness_;

    bool inverted_;
};

} // namespace matchbox