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

    enum Direction : int
    {
      DIR_NONE                     = 0b00000000,
      DIR_LEFT_TO_RIGHT            = 0b00000001,
      DIR_RIGHT_TO_LEFT            = 0b00000010,
      DIR_TOP_TO_BOTTOM            = 0b00000100,
      DIR_BOTTOM_TO_TOP            = 0b00001000,
      DIR_TOP_LEFT_TO_BOTTOM_RIGHT = 0b00010000,
      DIR_BOTTOM_LEFT_TO_TOP_RIGHT = 0b00100000,
      DIR_TOP_RIGHT_TO_BOTTOM_LEFT = 0b01000000,
      DIR_BOTTOM_RIGHT_TO_TOP_LEFT = 0b10000000,
      DIR_HORIZONTAL_VERTICAL      = 0b00001111,
      DIR_HORIZONTAL               = 0b00000011,
      DIR_ALL                      = 0b11111111,
    };

  public:

    Aggregator(std::shared_ptr<const MatchingCost> matching_cost);

    virtual ~Aggregator();

    std::shared_ptr<const MatchingCost> GetMatchingCost() const;

    int GetDirections() const;

    void SetDirections(int directions);

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

    int directions_;
};

} // namespace matchbox
