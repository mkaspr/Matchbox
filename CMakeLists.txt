cmake_minimum_required(VERSION 2.8)
project(Matchbox)

# find packages

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND REQUIRED_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)
list(APPEND REQUIRED_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/include)

find_package(CUDA REQUIRED)
list(APPEND REQUIRED_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
list(APPEND REQUIRED_LIBRARIES ${CUDA_LIBRARIES})

find_package(OpenCV REQUIRED)
list(APPEND REQUIRED_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
list(APPEND REQUIRED_LIBRARIES ${OpenCV_LIBRARIES})

find_package(GFlags REQUIRED)
list(APPEND REQUIRED_INCLUDE_DIRS ${GFlags_INCLUDE_DIRS})
list(APPEND REQUIRED_LIBRARIES ${GFlags_LIBRARIES})

find_package(GLog REQUIRED)
list(APPEND REQUIRED_INCLUDE_DIRS ${GLog_INCLUDE_DIRS})
list(APPEND REQUIRED_LIBRARIES ${GLog_LIBRARIES})

# set compiler flags

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=compute_50")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# define c++ configs

set(CPP_CONFIG_DIR include/matchbox)

set(CPP_CONFIGS
  ${CPP_CONFIG_DIR}/matchbox.h
)

# define c++ headers

set(CPP_HEADER_DIR include/matchbox)

set(CPP_HEADERS
  ${CPP_HEADER_DIR}/aggregate_cost.h
  ${CPP_HEADER_DIR}/aggregator.h
  ${CPP_HEADER_DIR}/device.h
  ${CPP_HEADER_DIR}/disparity_checker.h
  ${CPP_HEADER_DIR}/disparity_computer.h
  ${CPP_HEADER_DIR}/exception.h
  ${CPP_HEADER_DIR}/feature_extractor.h
  ${CPP_HEADER_DIR}/feature_map.h
  ${CPP_HEADER_DIR}/image.h
  ${CPP_HEADER_DIR}/matching_cost.h
  ${CPP_HEADER_DIR}/matcher.h
  ${CPP_HEADER_DIR}/median_filter.h
  ${CPP_HEADER_DIR}/time.h
)

# define c++ sources

set(CPP_SOURCE_DIR src)

set(CPP_SOURCES
)

# define cuda headers

set(CUDA_HEADER_DIR include/matchbox)

set(CUDA_HEADERS
  ${CUDA_HEADER_DIR}/util.cuh
)

# define cuda sources

set(CUDA_SOURCE_DIR src)

set(CUDA_SOURCES
  ${CUDA_SOURCE_DIR}/aggregate_cost.cu
  ${CUDA_SOURCE_DIR}/aggregator.cu
  ${CUDA_SOURCE_DIR}/disparity_checker.cu
  ${CUDA_SOURCE_DIR}/disparity_computer.cu
  ${CUDA_SOURCE_DIR}/feature_extractor.cu
  ${CUDA_SOURCE_DIR}/feature_extractor.cu
  ${CUDA_SOURCE_DIR}/feature_map.cu
  ${CUDA_SOURCE_DIR}/image.cu
  ${CUDA_SOURCE_DIR}/matching_cost.cu
  ${CUDA_SOURCE_DIR}/matcher.cu
  ${CUDA_SOURCE_DIR}/median_filter.cu
)

# build config files

foreach(CPP_HEADER ${CPP_HEADERS})
  string(REPLACE "include/" "" CPP_HEADER ${CPP_HEADER})
  set(MATCHBOX_INCLUDES "${MATCHBOX_INCLUDES}\n#include <${CPP_HEADER}>")
endforeach()

foreach(CPP_CONFIG ${CPP_CONFIGS})
  configure_file(${CPP_CONFIG}.in ${CPP_CONFIG} @ONLY)
endforeach()

# build library

include_directories(${REQUIRED_INCLUDE_DIRS})
cuda_add_library(matchbox SHARED ${CPP_SOURCES} ${CUDA_SOURCES})
target_link_libraries(matchbox ${REQUIRED_LIBRARIES})

# build applications

option(MATCHBOX_BUILD_APPS "Build applications" ON)

if(MATCHBOX_BUILD_APPS)
  add_subdirectory(apps)
endif()

# build unit tests

option(MATCHBOX_BUILD_TESTS "Build unit tests" ON)

if(MATCHBOX_BUILD_TESTS)
  add_subdirectory(tests)
endif()