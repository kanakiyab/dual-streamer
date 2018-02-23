# dual-streamer

Stream hi resolution and low resolution video/camera simultaneously on Jetson TX2/TX1 boards.

Build instructions:
```
git clone https://github.com/kanakiyab/dual-streamer.git
cd dual-streamer
mkdir build
cd build
cmake ..
make -j4
```

## Information

This will create two binaries in ```build/bin``` namely, ```gst-camera``` and ```gst-video```

To run the video file, an appropriate video file must be input. It should be of high resolution (e.g. 2560 x 1440).

```./gst-video --input /path/to/video.mp4```

To run the camera file, which is only configured for an RTSP stream for dual-streaming, just set the proper rtsp stream URL in [gstCamera.cpp L#341 and L#349](https://github.com/kanakiyab/dual-streamer/blob/1e7c600b6693c714043b1c33305232142367de18/camera/gstCamera.cpp#L341)

```./gst-camera```

Set the RTSP stream resolution to 2560x1440 or else, the resolution should be changed in the [gst-camera.cpp L#90](https://github.com/kanakiyab/dual-streamer/blob/1e7c600b6693c714043b1c33305232142367de18/camera/gst-camera/gst-camera.cpp#L90)


## Issue

When the ```gst-camera``` is executed, in the current state, it is unable to capture the frames at a high resolution. When the ```gst-video``` is executed, the framerate drops quite significantly and hence the pipoeline of running detection and tracking is slowed down.

---
This repo is just a slight modifications of the streamer code available at: https://github.com/dusty-nv/jetson-inference.
