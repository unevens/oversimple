cmake_minimum_required(VERSION 3.0.0)

project(oversimple-tests)

add_subdirectory("${CMAKE_SOURCE_DIR}/../" "${PROJECT_BINARY_DIR}/oversimple")
set(CMAKE_CXX_STANDARD 17)

add_executable(iir-design iir-design.cpp)
target_link_libraries(iir-design oversimple)

add_executable(ovr-test testing.cpp)
target_link_libraries(ovr-test oversimple)

if (WIN32)

    set(architecture "/arch:AVX") # comment out this line to build for SSE2
    target_compile_options(oversimple PUBLIC ${architecture})

elseif (NOT APPLE)

    set(architecture "-march=native") # replace this line to build for a specific architecture
    target_compile_options(oversimple PUBLIC ${architecture})

endif ()
