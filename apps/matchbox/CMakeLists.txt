add_executable(matchbox-bin matchbox.cpp)
target_link_libraries(matchbox-bin matchbox)
set_target_properties(matchbox-bin PROPERTIES OUTPUT_NAME matchbox)