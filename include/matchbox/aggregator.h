#pragma once

#include <memory>

namespace matchbox
{

class AggregateCost;
class MatchingCost;

class Aggregator
{
  public:

    Aggregator(std::shared_ptr<const MatchingCost> matching_cost);

    std::shared_ptr<const MatchingCost> GetMatchingCost() const;

    int GetDegree() const;

    void SetDegree(int degree);

    void Aggregate(AggregateCost& cost) const;

  protected:

    int GetPathCount() const;

    void ResizeCost(AggregateCost& cost) const;

    void AggregateMatching(AggregateCost& cost) const;

    void AggregateHorizontal(AggregateCost& cost) const;

    void AggregateVertical(AggregateCost& cost) const;

    void AggregateDiagonal(AggregateCost& cost) const;

  protected:

    std::shared_ptr<const MatchingCost> matching_cost_;

    int degree_;
};

} // namespace matchbox
