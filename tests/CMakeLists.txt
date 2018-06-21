enable_testing()

# find packages

list(APPEND REQUIRED_INCLUDE_DIRS ${CMAKE_SOURCE_DIR})
list(APPEND REQUIRED_INCLUDE_DIRS ${CMAKE_BINARY_DIR})
list(APPEND REQUIRED_LIBRARIES matchbox)

find_package(GTest REQUIRED)
list(APPEND REQUIRED_INCLUDE_DIRS ${GTEST_INCLUDE_DIRS})
list(APPEND REQUIRED_LIBRARIES ${GTEST_LIBRARIES})
list(APPEND REQUIRED_LIBRARIES pthread)

# define c++ sources

set(CPP_SOURCES
  exception_test.cpp
)

# define cuda sources

set(CUDA_SOURCES
  aggregate_cost_test.cu
  aggregator_test.cu
  disparity_checker_test.cu
  disparity_computer_test.cu
  feature_extractor_test.cu
  feature_map_test.cu
  image_test.cu
  matching_cost_test.cu
  matcher_test.cu
  median_filter_test.cu
  util_test.cu
)

# add executable

include_directories(${REQUIRED_INCLUDE_DIRS})
cuda_add_executable(tests tests.cpp ${CPP_SOURCES} ${CUDA_SOURCES})
target_link_libraries(tests ${REQUIRED_LIBRARIES})