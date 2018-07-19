#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace matchbox
{

class Image
{
  public:

    Image();

    Image(int w, int h);

    Image(const Image& image);

    Image(const std::string& file);

    Image& operator=(const Image& image);

    ~Image();

    int GetTotal() const;

    int GetWidth() const;

    int GetHeight() const;

    void SetSize(int w, int h);

    void Load(const cv::Mat& image);

    void Load(const std::string& file);

    void Save(const std::string& file) const;

    const uint8_t* GetData() const;

    uint8_t* GetData();

  protected:

    int width_;

    int height_;

    uint8_t* data_;
};

class DepthImage
{
  public:

    DepthImage();

    DepthImage(int w, int h);

    DepthImage(const DepthImage& image);

    DepthImage& operator=(const DepthImage& image);

    ~DepthImage();

    int GetBytes() const;

    int GetTotal() const;

    int GetWidth() const;

    int GetHeight() const;

    void SetSize(int w, int h);

    void Save(const std::string& file) const;

    const float* GetData() const;

    float* GetData();

  protected:

    int width_;

    int height_;

    float* data_;
};

} // namespace matchbox