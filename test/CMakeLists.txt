cmake_minimum_required(VERSION 3.0.0)

project(oversimple-tests)

set(CMAKE_CXX_STANDARD 17)

include_directories(../)
include_directories(../r8brain)
include_directories(../avec)
include_directories(../avec/vectorclass)

file(GLOB src ../r8brain/r8bbase.cpp ../r8brain/pffft_double/pffft_double.c ../oversimple/FirOversampling.cpp)

add_executable(iir-design ${src} iir-design.cpp)

if(WIN32)

add_executable(ovr-test-avx ${src} testing.cpp)
add_executable(ovr-test-sse2 ${src} testing.cpp)

target_compile_options (ovr-test-avx PRIVATE /arch:AVX)
target_compile_options (ovr-test-sse2 PRIVATE /arch:SSE2)
target_compile_definitions(ovr-test-avx PRIVATE R8B_PFFFT_DOUBLE=1)
target_compile_definitions(ovr-test-sse2 PRIVATE R8B_PFFFT_DOUBLE=1)

endif(WIN32)

if(UNIX)

add_executable(ovr-test-native ${src} testing.cpp)
add_executable(ovr-test ${src} testing.cpp)

target_compile_options (ovr-test-native PRIVATE -march=native)
target_compile_definitions(ovr-test-native PRIVATE R8B_PFFFT_DOUBLE=1)
target_compile_definitions(ovr-test PRIVATE R8B_PFFFT=1)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(ovr-test-native Threads::Threads)
target_link_libraries(ovr-test Threads::Threads)
target_link_libraries(iir-design Threads::Threads)
endif(UNIX)
