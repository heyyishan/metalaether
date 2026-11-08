cmake_minimum_required(VERSION 3.24)
project(aether_metal VERSION 0.1.0 LANGUAGES CXX OBJCXX)

if(NOT APPLE)
  message(FATAL_ERROR "Aether requires macOS and Apple Metal.")
endif()
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 23)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum macOS version")

find_package(Python COMPONENTS Interpreter Development.Module QUIET)
find_package(pybind11 CONFIG QUIET)

set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/aether.air)
set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)
add_custom_command(OUTPUT ${METAL_LIB}
  COMMAND xcrun -sdk macosx metal -std=macos-metal3.1 -O3 -c ${CMAKE_CURRENT_SOURCE_DIR}/metal/kernels.metal -o ${METAL_AIR}
  COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
  DEPENDS metal/kernels.metal VERBATIM)
add_custom_target(aether_metallib ALL DEPENDS ${METAL_LIB})

add_library(aether_core STATIC src/metal_runtime.mm src/model.cpp src/tokenizer.cpp src/engine.cpp)
add_dependencies(aether_core aether_metallib)
target_include_directories(aether_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(aether_core PRIVATE AETHER_METALLIB_PATH="${METAL_LIB}")
target_link_libraries(aether_core PUBLIC "-framework Foundation" "-framework Metal" "-framework MetalPerformanceShaders")
target_compile_options(aether_core PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>" "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")

add_executable(aether examples/generate.cpp)
target_link_libraries(aether PRIVATE aether_core)
if(pybind11_FOUND)
  pybind11_add_module(_aether python/bindings.cpp)
  target_link_libraries(_aether PRIVATE aether_core)
  install(TARGETS _aether LIBRARY DESTINATION aether)
endif()
install(TARGETS aether RUNTIME DESTINATION bin)
