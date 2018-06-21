#include <chrono>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <matchbox/matchbox.h>

DEFINE_string(left, "", "path to left image");
DEFINE_string(right, "", "path to right image");
DEFINE_int32(max_disp, 128, "max disparity");
DEFINE_int32(degree, 3, "aggregate cost degree");
DEFINE_int32(iters, 10, "number of iterations");

using namespace matchbox;

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "Parsing flags...";
  MATCHBOX_ASSERT_MSG(!FLAGS_left.empty(), "left image required");
  MATCHBOX_ASSERT_MSG(!FLAGS_right.empty(), "right image required");

  LOG(INFO) << "Reading left image...";
  std::shared_ptr<Image> left_image;
  left_image = std::make_shared<Image>();
  left_image->Load(FLAGS_left);

  LOG(INFO) << "Reading right image...";
  std::shared_ptr<Image> right_image;
  right_image = std::make_shared<Image>();
  right_image->Load(FLAGS_right);

  LOG(INFO) << "Extracting left features...";
  std::shared_ptr<FeatureMap> left_features;
  left_features = std::make_shared<FeatureMap>();
  FeatureExtractor left_extractor(left_image);
  left_extractor.Extract(*left_features);

  LOG(INFO) << "Extracting right features...";
  std::shared_ptr<FeatureMap> right_features;
  right_features = std::make_shared<FeatureMap>();
  FeatureExtractor right_extractor(right_image);
  right_extractor.Extract(*right_features);

  LOG(INFO) << "Computing matching cost...";
  std::shared_ptr<MatchingCost> matching_cost;
  matching_cost = std::make_shared<MatchingCost>();
  Matcher matcher(left_features, right_features);
  matcher.SetMaxDisparity(FLAGS_max_disp);
  matcher.Match(*matching_cost);

  LOG(INFO) << "Computing aggregate cost...";
  std::shared_ptr<AggregateCost> aggregate_cost;
  aggregate_cost = std::make_shared<AggregateCost>();
  Aggregator aggregator(matching_cost);
  aggregator.SetDegree(FLAGS_degree);
  aggregator.Aggregate(*aggregate_cost);

  LOG(INFO) << "Computing left disparities...";
  std::shared_ptr<Image> left_disparities;
  left_disparities = std::make_shared<Image>();
  DisparityComputer left_computer(aggregate_cost);
  left_computer.SetInverted(false);
  left_computer.Compute(*left_disparities);

  LOG(INFO) << "Computing right disparities...";
  std::shared_ptr<Image> right_disparities;
  right_disparities = std::make_shared<Image>();
  DisparityComputer right_computer(aggregate_cost);
  right_computer.SetInverted(true);
  right_computer.Compute(*right_disparities);

  LOG(INFO) << "Filtering left disparities...";
  std::shared_ptr<Image> left_filtered;
  left_filtered = std::make_shared<Image>();
  MedianFilter left_filter(left_disparities);
  left_filter.Filter(*left_filtered);

  LOG(INFO) << "Filtering right disparities...";
  std::shared_ptr<Image> right_filtered;
  right_filtered = std::make_shared<Image>();
  MedianFilter right_filter(right_disparities);
  right_filter.Filter(*right_filtered);

  LOG(INFO) << "Checking disparities...";
  DisparityChecker checker(left_filtered, right_filtered);
  checker.SetMode(DisparityChecker::MODE_CHECK_LEFT);
  checker.SetMaxDifference(1);
  checker.Check();

  const int iters = FLAGS_iters;

  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  for (int i = 0; i < iters; ++i)
  {
    left_extractor.Extract(*left_features);
    right_extractor.Extract(*right_features);
    matcher.Match(*matching_cost);
    aggregator.Aggregate(*aggregate_cost);
    left_computer.Compute(*left_disparities);
    right_computer.Compute(*right_disparities);
    left_filter.Filter(*left_filtered);
    right_filter.Filter(*right_filtered);
    checker.Check();
  }

  std::chrono::steady_clock::time_point stop =
      std::chrono::steady_clock::now();

  std::chrono::duration<double> delta = stop - start;
  const double time = std::chrono::duration_cast<std::chrono::microseconds>(delta).count();
  LOG(INFO) << "Time per frame: " << time / (1000000 * iters);
  LOG(INFO) << "Frames per sec: " << 1.0 / (time / (1000000 * iters));
  // current best: ~14 fps

  LOG(INFO) << "Success";
  return 0;
}