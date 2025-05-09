set(CMAKE_C_COMPILER /opt/homebrew/opt/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /opt/homebrew/opt/llvm/bin/clang++)

cmake_minimum_required(VERSION 3.14)

project(abovo)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

include_directories(include)

file(GLOB_RECURSE SOURCES src/*.cpp)

pybind11_add_module(abovo pybind/bindings.cpp ${SOURCES})

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(abovo PRIVATE OpenMP::OpenMP_CXX)
endif()

