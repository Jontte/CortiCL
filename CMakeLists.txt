project(corticl)

cmake_minimum_required (VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(OpenCL REQUIRED)
find_package(SDL2)

if (${SDL2_FOUND})
  include_directories(${SDL2_INCLUDE_DIRS})
endif()

include_directories(${OPENCL_INCLUDE_DIRS})
include_directories(${PROJECT_BINARY_DIR})

set(LIBS ${LIBS} ${SDL2_LIBRARIES} ${OPENCL_LIBRARIES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -Wno-deprecated-declarations -g -pthread -std=c++0x")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")

add_custom_command(
	PRE_BUILD
	OUTPUT ${PROJECT_BINARY_DIR}/spatial.cl.h
	COMMAND ${CMAKE_COMMAND} -D SOURCE=${PROJECT_SOURCE_DIR}/src/cl/spatial.cl -D DESTINATION=${PROJECT_BINARY_DIR}/spatial.cl.h -P ${CMAKE_SOURCE_DIR}/cmake/stringify.cmake
	DEPENDS ${PROJECT_SOURCE_DIR}/src/cl/spatial.cl
)

add_custom_command(
	PRE_BUILD
	OUTPUT ${PROJECT_BINARY_DIR}/temporal.cl.h
	COMMAND ${CMAKE_COMMAND} -D SOURCE=${PROJECT_SOURCE_DIR}/src/cl/temporal.cl -D DESTINATION=${PROJECT_BINARY_DIR}/temporal.cl.h -P ${CMAKE_SOURCE_DIR}/cmake/stringify.cmake
	DEPENDS ${PROJECT_SOURCE_DIR}/src/cl/temporal.cl
)

add_library(corticl STATIC
	src/clregion.cpp
	src/clspatial.cpp
	src/cltemporal.cpp
	src/clargs.cpp
	src/cltopology.cpp
	src/clcontext.cpp
	${PROJECT_BINARY_DIR}/spatial.cl.h
	${PROJECT_BINARY_DIR}/temporal.cl.h
)

if (${SDL2_FOUND})
	add_executable(cldemo
	src/demo/cldemo.cpp
	src/demo/cldemo1.cpp
	src/demo/cldemo2.cpp
	src/demo/cldemo3.cpp
	src/demo/sensor.cpp)

	target_link_libraries(cldemo corticl SDL2 GL ${OPENCL_LIBRARIES})
else()
	message(STATUS "SDL2 not found, not building cldemo")
endif()

add_executable(basic src/demo/basic.cpp)
target_link_libraries(basic corticl ${OPENCL_LIBRARIES})

