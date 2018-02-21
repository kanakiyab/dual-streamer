
file(GLOB gstVideoSources *.cpp)
file(GLOB gstVideoIncludes *.h)

add_executable(gst-video ${gstVideoSources})
target_link_libraries(gst-video streamer)
