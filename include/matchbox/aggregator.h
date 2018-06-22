#pragma once

#include <memory>
#include <matchbox/device.h>

namespace matchbox
{

class AggregateCost;
class MatchingCost;

class Aggregator
{
  protected:

    static const int stream_count = 4;

  public:

    Aggregator(std::shared_ptr<const MatchingCost> matching_cost);

    virtual ~Aggregator();

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

  private:

    void Initialize();

  protected:

    std::shared_ptr<const MatchingCost> matching_cost_;

    cudaStream_t streams[stream_count];

    int degree_;
};

} // namespace matchbox
