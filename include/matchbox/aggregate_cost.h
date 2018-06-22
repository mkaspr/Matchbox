#pragma once

#include <cstdint>
#include <cstddef>

namespace matchbox
{

class AggregateCost
{
  public:

    AggregateCost();

    AggregateCost(int w, int h, int d, int p);

    AggregateCost(const AggregateCost& cost);

    AggregateCost& operator=(const AggregateCost& cost);

    ~AggregateCost();

    size_t GetBytes() const;

    int GetTotal() const;

    int GetWidth() const;

    int GetHeight() const;

    int GetDepth() const;

    int GetPaths() const;

    void SetSize(int w, int h, int d, int p);

    const uint8_t* GetData() const;

    uint8_t* GetData();

    void Clear();

  protected:

    int width_;

    int height_;

    int depth_;

    int paths_;

    uint8_t* data_;
};

} // namespace matchbox