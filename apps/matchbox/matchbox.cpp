#include <gflags/gflags.h>
#include <glog/logging.h>
#include <matchbox/matchbox.h>

using namespace matchbox;

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Success";
  return 0;
}