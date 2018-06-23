#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <matchbox/matchbox.h>

#include <thrust/device_ptr.h>

DEFINE_string(left, "", "path to left image");
DEFINE_string(right, "", "path to right image");
DEFINE_int32(max_disp, 128, "max disparity");
DEFINE_int32(degree, 3, "aggregate cost degree");
DEFINE_bool(check, true, "perform disparity check");
DEFINE_bool(check_both, false, "perform disparity check on both frames");
DEFINE_bool(filter, true, "perform median filter before disparity check");
DEFINE_double(uniqueness, 0.95, "uniqueness threshold for disparity computer");

using namespace matchbox;

DisparityChecker::Mode GetCheckerMode()
{
  return FLAGS_check_both ?
      DisparityChecker::MODE_CHECK_BOTH :
      DisparityChecker::MODE_CHECK_LEFT;
}

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::shared_ptr<Image> left_result;
  std::shared_ptr<Image> right_result;

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
  left_computer.SetUniqueness(FLAGS_uniqueness);
  left_computer.SetInverted(false);
  left_computer.Compute(*left_disparities);
  left_result = left_disparities;

  if (FLAGS_filter)
  {
    LOG(INFO) << "Filtering left disparities...";
    left_result = std::make_shared<Image>();
    MedianFilter left_filter(left_disparities);
    left_filter.Filter(*left_result);
  }

  if (FLAGS_check)
  {
    LOG(INFO) << "Computing right disparities...";
    std::shared_ptr<Image> right_disparities;
    right_disparities = std::make_shared<Image>();
    DisparityComputer right_computer(aggregate_cost);
    right_computer.SetUniqueness(FLAGS_uniqueness);
    right_computer.SetInverted(true);
    right_computer.Compute(*right_disparities);
    right_result = right_disparities;

    if (FLAGS_filter)
    {
      LOG(INFO) << "Filtering right disparities...";
      right_result = std::make_shared<Image>();
      MedianFilter right_filter(right_disparities);
      right_filter.Filter(*right_result);
    }

    LOG(INFO) << "Checking disparities...";
    DisparityChecker checker(left_result, right_result);
    checker.SetMode(GetCheckerMode());
    checker.SetMaxDifference(1);
    checker.Check();
  }

  // TODO: clean up
  const int w = left_disparities->GetWidth();
  const int h = left_disparities->GetHeight();
  const uint8_t* data;
  cv::Mat result;

  data = left_result->GetData();
  result = cv::Mat(h, w, CV_8UC1);
  CUDA_DEBUG(cudaMemcpy(result.data, data, w * h, cudaMemcpyDeviceToHost));
  cv::imshow("Left Disparities", 3 * result);

  if (FLAGS_check && FLAGS_check_both)
  {
    data = right_result->GetData();
    result = cv::Mat(h, w, CV_8UC1);
    CUDA_DEBUG(cudaMemcpy(result.data, data, w * h, cudaMemcpyDeviceToHost));
    cv::imshow("Right Disparities", 3 * result);
  }

  while (cv::waitKey(10) != 27);

  LOG(INFO) << "Success";
  return 0;
}