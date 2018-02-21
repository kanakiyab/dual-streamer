/*
 * This file is put together from the original work by Nvidia author dusty-nv
 * and the work by omaralvarez for RTSP Pipeline. Their codes are available 
 * on their respective GitHub profiles under the repo jetson-inference.
 * The original Nvidia license information is retained as is.
 * Author: Bhargav Kanakiya
 * email: bhargav@automot.us
 */

/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#ifndef __GSTREAMER_VIDEO_H__
#define __GSTREAMER_VIDEO_H__

#include <gst/gst.h>
#include <string>

struct _GstAppSink;
class QWaitCondition;
class QMutex;

#include "cuda_runtime.h"

/**
 * gstreamer video pipeline using gst-launch
 */
class gstVideo {
    public:
        
        static gstVideo* Create(const char* videoPath, bool downscaled);
        static gstVideo* Create(const char* videoPath, uint32_t width,
                                uint32_t height, bool downscaled);
        static gstVideo* Create(const char* videoPath, uint32_t width,
                                uint32_t height, uint32_t depth, bool downscaled);
                                
        // Destroy
        ~gstVideo();
        
        // Start/Stop the stream
        bool Open();
        void Close();
        
        // Capture YUV (NV12)
        bool Capture(void** cpu, void** cuda, unsigned long timeout=ULONG_MAX);
        
        // Takes in the YUV-NV12 CUDA image and converts to float4 RGBA 
        // with pixel intensity between 0-255. Done on GPU usin CUDA
        bool ConvertRGBA(void* input, void** output, bool zeroCopy);

        // number of frames
        inline uint32_t GetFrameCount() { return mFrameCount; }

        // Image dimensions
        inline uint32_t GetWidth() const        { return mWidth; }
        inline uint32_t GetHeight() const       { return mHeight; }
        inline uint32_t GetPixelDepth() const   { return mDepth; }
        inline uint32_t GetSize() const         { return mSize; }
        
        // Default resolution, unless otherwise specified during Create
        static const uint32_t DefaultWidth = 640;
        static const uint32_t DefaultHeight = 360;
        static const uint32_t DefaultDepth = 12;
    
    private:
        static void onEOS(_GstAppSink* sink, void* user_data);
        static GstFlowReturn onPreroll(_GstAppSink* sink, void* user_data);
        static GstFlowReturn onBuffer(_GstAppSink* sink, void* user_data);
        
        gstVideo();
        
        bool init();
        void checkMsgBus();
        void checkBuffer();
        // no buildLaunchStr() because we have a member for storing it below
        
        _GstBus* mBus;
        _GstAppSink* mAppSink;
        _GstElement* mPipeline;
        
        std::string mLaunchStr;
        
        uint32_t mWidth;
        uint32_t mHeight;
        uint32_t mDepth;
        uint32_t mSize;
        uint32_t mFrameCount;
        
        static const uint32_t NUM_RINGBUFFERS = 8;

        void* mRingBufferCPU[NUM_RINGBUFFERS];
        void* mRingBufferGPU[NUM_RINGBUFFERS];
        
        QWaitCondition* mWaitEvent;
        
        QMutex* mWaitMutex;
        QMutex* mRingMutex;
        
        uint32_t mLatestRGBA;
        uint32_t mLatestRingbuffer;
        bool mLatestReceived;
        
        void* mRGBA[NUM_RINGBUFFERS];
};

#endif
