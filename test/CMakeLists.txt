cmake_minimum_required(VERSION 3.16)
project(oversimple-tests CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Driven by the run-tests.sh script per-arch. The script invokes cmake with
# CMAKE_OSX_ARCHITECTURES set explicitly (e.g. "arm64" or "x86_64") so the
# build dir is single-arch and tests can actually run via `arch -<slice>`.
add_subdirectory("${CMAKE_SOURCE_DIR}/../" "${PROJECT_BINARY_DIR}/oversimple")

add_executable(iir-design iir-design.cpp)
target_link_libraries(iir-design oversimple)

add_executable(oversimple-test testing.cpp)
target_link_libraries(oversimple-test oversimple)
target_include_directories(oversimple-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

if(UNIX AND NOT APPLE)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif()

enable_testing()
add_test(NAME oversimple-test COMMAND oversimple-test)
