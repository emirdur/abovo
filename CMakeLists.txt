cmake_minimum_required(VERSION 3.14)

if(APPLE AND NOT DEFINED ENV{CI})
    if(EXISTS "/opt/homebrew/opt/llvm/bin/clang")
        set(CMAKE_C_COMPILER /opt/homebrew/opt/llvm/bin/clang)
        set(CMAKE_CXX_COMPILER /opt/homebrew/opt/llvm/bin/clang++)
    endif()
endif()

project(abovo)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

include_directories(include)

file(GLOB_RECURSE SOURCES src/*.cpp)

pybind11_add_module(abovo pybind/bindings.cpp ${SOURCES})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(abovo PRIVATE OpenMP::OpenMP_CXX)
endif()