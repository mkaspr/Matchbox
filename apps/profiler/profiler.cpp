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
DEFINE_bool(check, true, "perform consistency check");

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

  LOG(INFO) << "Filtering left disparities...";
  std::shared_ptr<Image> left_filtered;
  left_filtered = std::make_shared<Image>();
  MedianFilter left_filter(left_disparities);
  left_filter.Filter(*left_filtered);

  std::shared_ptr<Image> right_filtered;
  std::shared_ptr<Image> right_disparities;
  std::shared_ptr<DisparityComputer> right_computer;
  std::shared_ptr<MedianFilter> right_filter;
  std::shared_ptr<DisparityChecker> checker;

  if (FLAGS_check)
  {
    LOG(INFO) << "Computing right disparities...";
    right_disparities = std::make_shared<Image>();
    right_computer = std::make_shared<DisparityComputer>(aggregate_cost);
    right_computer->SetInverted(true);
    right_computer->Compute(*right_disparities);

    LOG(INFO) << "Filtering right disparities...";
    right_filtered = std::make_shared<Image>();
    right_filter = std::make_shared<MedianFilter>(right_disparities);
    right_filter->Filter(*right_filtered);

    LOG(INFO) << "Checking disparities...";
    checker = std::make_shared<DisparityChecker>(left_filtered, right_filtered);
    checker->SetMode(DisparityChecker::MODE_CHECK_LEFT);
    checker->SetMaxDifference(1);
    checker->Check();
  }

  cudaDeviceSynchronize();
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
    left_filter.Filter(*left_filtered);

    if (FLAGS_check)
    {
      right_computer->Compute(*right_disparities);
      right_filter->Filter(*right_filtered);
      checker->Check();
    }

    cudaDeviceSynchronize();
  }

  std::chrono::steady_clock::time_point stop =
      std::chrono::steady_clock::now();

  std::chrono::duration<double> delta = stop - start;
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(delta);
  const double time = duration.count() / (1000000.0 * iters);
  LOG(INFO) << "Frames per sec: " << 1.0 / time;
  LOG(INFO) << "Time per frame: " << time;
  // RELEASE: current best w/o check: ~36.0 fps
  // RELEASE: current best w/ check:  ~17.5 fps

  LOG(INFO) << "Success";
  return 0;
}