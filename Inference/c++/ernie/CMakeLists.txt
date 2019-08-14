CMAKE_MINIMUM_REQUIRED(VERSION 3.2)

project(inference_demo CXX C)
set(DEMO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DEMO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(USE_SHARED OFF)
set(USE_GPU ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(PADDLE_ROOT ${DEMO_SOURCE_DIR}/fluid_inference)
find_package(Fluid)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(PADDLE_FLUID_FOUND)
  add_executable(inference inference.cc)
  target_link_libraries(inference
      ${PADDLE_LIBRARIES}
      ${PADDLE_THIRD_PARTY_LIBRARIES}
      rt dl pthread gomp)
else()
  message(FATAL_ERROR "Cannot find PaddlePaddle Fluid under ${PADDLE_ROOT}")
endif()
