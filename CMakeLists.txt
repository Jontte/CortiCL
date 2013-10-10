project(corticl)

cmake_minimum_required (VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(OpenCL REQUIRED)
find_package(SDL2 REQUIRED)

include_directories(${SDL2_INCLUDE_DIRS})
include_directories(${OPENCL_INCLUDE_DIRS})
set(LIBS ${LIBS} ${SDL2_LIBRARIES} ${OPENCL_LIBRARIES})

#find_package(OpenMP)
#if (OPENMP_FOUND)
#	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()

add_definitions(-std=c++11 -DLINUX -Wno-deprecated-declarations)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -g -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")

add_executable(cldemo src/demo/cldemo.cpp src/demo/sensor.cpp src/clregion.cpp src/clspatial.cpp src/cltemporal.cpp src/clargs.cpp)

target_link_libraries(cldemo SDL2 GL ${OPENCL_LIBRARIES})

file(COPY ${PROJECT_SOURCE_DIR}/src/cl/spatial.cl DESTINATION cl)
file(COPY ${PROJECT_SOURCE_DIR}/src/cl/temporal.cl DESTINATION cl)