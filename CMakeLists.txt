cmake_minimum_required(VERSION 2.8)
project(Matchbox)

# find packages

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND REQUIRED_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)
list(APPEND REQUIRED_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/include)

find_package(CUDA REQUIRED)
list(APPEND REQUIRED_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
list(APPEND REQUIRED_LIBRARIES ${CUDA_LIBRARIES})

# set compiler flags

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# define c++ configs

set(CPP_CONFIG_DIR matchbox/include)

set(CPP_CONFIGS
  ${CPP_CONFIG_DIR}/matchbox.h
)

# define c++ headers

set(CPP_HEADER_DIR matchbox/include)

set(CPP_HEADERS
  ${CPP_HEADER_DIR}/exception.h
)

# define c++ sources

set(CPP_SOURCE_DIR src)

set(CPP_SOURCES
  ${CPP_SOURCE_DIR}/dummy.cpp
)

# define cuda headers

set(CUDA_HEADER_DIR matchbox/include)

set(CUDA_HEADERS
)

# define cuda sources

set(CUDA_SOURCE_DIR src)

set(CUDA_SOURCES
)

# build library

include_directories(${REQUIRED_INCLUDE_DIRS})
add_library(matchbox SHARED ${CPP_SOURCES} ${CUDA_SOURCES})
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