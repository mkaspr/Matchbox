find_package(HAL)

if(HAL_FOUND)

  include_directories(matchbox_live ${HAL_INCLUDE_DIRS})
  add_executable(matchbox_live matchbox_live.cpp)
  target_link_libraries(matchbox_live ${HAL_LIBRARIES})
  target_link_libraries(matchbox_live matchbox)

endif()