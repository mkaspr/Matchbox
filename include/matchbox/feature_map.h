#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace matchbox
{

class FeatureMap
{
  public:

    FeatureMap();

    FeatureMap(int w, int h);

    FeatureMap(const FeatureMap& map);

    FeatureMap& operator=(const FeatureMap& map);

    ~FeatureMap();

    size_t GetBytes() const;

    int GetTotal() const;

    int GetWidth() const;

    int GetHeight() const;

    void SetSize(int w, int h);

    const uint64_t* GetData() const;

    uint64_t* GetData();

  protected:

    int width_;

    int height_;

    uint64_t* data_;
};

} // namespace matchbox