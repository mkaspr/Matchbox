#pragma once

#include <cstdint>

namespace matchbox
{

class MatchingCost
{
  public:

    MatchingCost();

    MatchingCost(int w, int h, int d);

    MatchingCost(const MatchingCost& cost);

    MatchingCost& operator=(const MatchingCost& cost);

    ~MatchingCost();

    int GetTotal() const;

    int GetWidth() const;

    int GetHeight() const;

    int GetDepth() const;

    void SetSize(int w, int h, int d);

    const uint8_t* GetData() const;

    uint8_t* GetData();

  protected:

    int width_;

    int height_;

    int depth_;

    uint8_t* data_;
};

} // namespace matchbox