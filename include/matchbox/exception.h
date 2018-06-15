#pragma once

#include <exception>
#include <string>

#define MATCHBOX_THROW(text)                                                   \
  throw ::matchbox::Exception(__LINE__, __FILE__, text)

#define MATCHBOX_ASSERT_MSG(cond, text)                                        \
  if (!(cond)) MATCHBOX_THROW(text)

#define MATCHBOX_ASSERT(cond)                                                  \
  if (!(cond)) MATCHBOX_THROW("assertion failed: " #cond)

#ifdef NDEBUG
#define MATCHBOX_DEBUG_MSG(cond, text)
#define MATCHBOX_DEBUG(cond)
#else
#define MATCHBOX_DEBUG_MSG MATCHBOX_ASSERT_MSG
#define MATCHBOX_DEBUG MATCHBOX_ASSERT
#endif

namespace matchbox
{

class Exception : public std::exception
{
  public:

    Exception(int line, const std::string& file, const std::string& text) :
      line_(line),
      file_(file),
      text_(text)
    {
      Initialize();
    }

    inline int line() const
    {
      return line_;
    }

    inline const std::string& file() const
    {
      return file_;
    }

    inline const std::string& text() const
    {
      return text_;
    }

    inline const char* what() const throw() override
    {
      return what_.c_str();
    }

  private:

    void Initialize()
    {
      what_ = file_ + "(" + std::to_string(line_) + "): " + text_;
    }

  protected:

    int line_;

    std::string file_;

    std::string text_;

    std::string what_;
};

} // namespace matchbox