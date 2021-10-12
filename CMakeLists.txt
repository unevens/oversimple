# add this project to your cmake project using add_subdirectory
# after that you can optionally specify the architecture to build by calling something like
# target_compile_options(oversimple PUBLIC /arch:AVX)
# see test/CMakeLists.txt for an example

cmake_minimum_required(VERSION 3.14.0)

# on MacOS, by default unplug will build an universal binary. you can set oversimple_override_macos_arch to the architecture
# you want to build by uncommenting one of the next two lines. It can be useful because Compiler Explorer does not support
# universal binaries.
# Note: CMake sometimes does not update this when you reload the project without deleting the build folder manually.
set(oversimple_override_macos_arch "")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_library(oversimple STATIC)

target_include_directories(oversimple PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(oversimple PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/r8brain")
target_include_directories(oversimple PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/avec")
target_include_directories(oversimple PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/avec/vectorclass")
target_sources(oversimple PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/r8brain/r8bbase.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/r8brain/pffft_double/pffft_double.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/oversimple/FirOversampling.cpp")
target_compile_definitions(oversimple PUBLIC R8B_PFFFT_DOUBLE=1)

if (UNIX)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(oversimple Threads::Threads)
    #universal binary if building on arm64
    if (APPLE)
        if (oversimple_override_macos_arch STREQUAL "arm64")
            set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "")
            message(STATUS "Supported architectures: arm64 (forced by user)")
        elseif (oversimple_override_macos_arch STREQUAL "x86_64")
            set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "")
            message(STATUS "Supported architectures: x86_64 (forced by user)")
        else ()
            execute_process(
                    COMMAND uname -m
                    RESULT_VARIABLE result
                    OUTPUT_VARIABLE MACHINE_ARCHITECTURE
                    OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            if (MACHINE_ARCHITECTURE STREQUAL "arm64")
                set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "")
                message(STATUS "Supported architectures: x86_64, arm64")
            else ()
                message(STATUS "Supported architectures: ${MACHINE_ARCHITECTURE}")
            endif ()
        endif ()
    else ()
        target_compile_options(ovr-test-native PUBLIC -march=native)
    endif ()
endif (UNIX)

if (WIN32)
    #default to sse2 on windows, can be overridden after
    target_compile_options(oversimple PUBLIC /arch:SSE2)
endif (WIN32)