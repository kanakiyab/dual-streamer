cmake_minimum_required(VERSION 3.2)

project(streamer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")


# if this is the first time running cmake, perform pre-build dependency install script (or if the user manually triggers re-building the dependencies)
if( ${BUILD_DEPS} )
    message("Launching pre-build dependency installer script...")

    execute_process(COMMAND sh ../CMakePrebuild.sh
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                RESULT_VARIABLE PREBUILD_SCRIPT_RESULT)

    set(BUILD_DEPS "NO" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed." FORCE)
    message("Finished installing dependencies")
endif()

# setup project output paths
set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR})
set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)

file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)

message("-- output path:  ${PROJECT_OUTPUT_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)

# setup QT4
find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

# setup CUDA
find_package(CUDA)

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}; 
    -O3 
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_62,code=sm_62
)

include_directories(${PROJECT_INCLUDE_DIR})
include_directories(/usr/include/gstreamer-1.0
                    /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include
                    /usr/include/glib-2.0
                    /usr/include/libxml2
                    /usr/lib/aarch64-linux-gnu/glib-2.0/include)

file(GLOB streamerSources camera/*.cpp video/*.cpp display/*.cpp cuda/*.cu *.cpp)
file(GLOB streamerIncludes camera/*.h video/*.h display/*.h cuda/*.h *.h)

cuda_add_library(streamer ${streamerSources})
target_link_libraries(streamer Qt4::QtGui GL GLEW gstreamer-1.0 gstapp-1.0 gobject-2.0 glib-2.0 X11)

# transfer all headers to the include directory
foreach(include ${streamerIncludes})
	message("-- Copying ${include}")
	configure_file(${include} ${PROJECT_INCLUDE_DIR} COPYONLY)
endforeach()

add_subdirectory(camera/gst-camera)
add_subdirectory(video/gst-video)
