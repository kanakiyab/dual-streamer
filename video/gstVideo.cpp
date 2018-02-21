 /**
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
 
#include "gstVideo.h"
#include "gstUtility.h"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <sstream> 
#include <unistd.h>
#include <string.h>

#include <QMutex>
#include <QWaitCondition>

#include "cudaMappedMemory.h"
#include "cudaYUV.h"
#include "cudaRGB.h"

//#include "tensorNet.h"

// Constructor
gstVideo::gstVideo() {
    mAppSink = NULL;
    mBus = NULL;
    mPipeline = NULL;
    
    mWidth = 0;
    mHeight = 0;
    mDepth = 0;
    mSize = 0;
    mFrameCount = 0;
    
    mWaitEvent = new QWaitCondition();
    mWaitMutex = new QMutex();
    mRingMutex = new QMutex();
    
    mLatestRGBA = 0;
    mLatestRingbuffer = 0;
    mLatestReceived = false;
    
    for(uint32_t n = 0; n < NUM_RINGBUFFERS; n++) {
        mRingBufferCPU[n] = NULL;
        mRingBufferGPU[n] = NULL;
        mRGBA[n] = NULL;
    }
}

// Desctructor
gstVideo::~gstVideo() {
    
}

// ConvertRGBA
bool gstVideo::ConvertRGBA(void* input, void** output, bool zeroCopy) {
    if(!input || !output) 
        return false;
        
    if(!mRGBA[0]) {
        const size_t size = mWidth * mHeight * sizeof(float4);
        for(uint32_t n = 0; n < NUM_RINGBUFFERS; n++) {
            if (zeroCopy) {
                void* cpuPtr = NULL;
                void* gpuPtr = NULL;
                
                if (!cudaAllocMapped(&cpuPtr, &gpuPtr, size)) {
                    printf(LOG_CUDA "gstVideo -- failed to acllocate ");
                    printf("zeroCopy memory for %ux%xu RGBA texture\n", 
                        mWidth, mHeight);
                    return false;
                }
                
                if (cpuPtr != gpuPtr) {
                    printf(LOG_CUDA "gstVideo -- zeroCopy memory has ");
                    printf("different pointers, please use a UVA-compatible ");
                    printf("GPU\n");
                }
                mRGBA[n] = gpuPtr;
            } else {
                if(CUDA_FAILED(cudaMalloc(&mRGBA[n], 
                                mWidth*mHeight*sizeof(float4)))) {
                    printf(LOG_CUDA "gstVideo -- failed to allocate ");
                    printf("memory for %ux%u RGB texture\n", mWidth, mHeight);
                    return false;
                }
            }
        }
        printf(LOG_CUDA "gstreamer video -- allocated ");
        printf("%u RGBA ringbuffers\n", NUM_RINGBUFFERS);
    }
    
    if(mDepth == 12) {
        //NV12
        if(CUDA_FAILED(cudaNV12ToRGBAf((uint8_t*)input, 
                        (float4*)mRGBA[mLatestRGBA], mWidth, mHeight)))
            return false;
    } else {
        if(CUDA_FAILED(cudaRGBToRGBAf((uchar3*)input, 
                        (float4*)mRGBA[mLatestRGBA], mWidth, mHeight)))
            return false;
    }
    
    *output = mRGBA[mLatestRGBA];
    mLatestRGBA = (mLatestRGBA + 1) % NUM_RINGBUFFERS;
    return true;
}

// onEOS
void gstVideo::onEOS(_GstAppSink* sink, void* user_data) {
    printf(LOG_GSTREAMER "gstreamer decoder onEOS\n");
}

// onPreroll
GstFlowReturn gstVideo::onPreroll(_GstAppSink* sink, void* user_data) {
    printf(LOG_GSTREAMER "gstreamer decoder onPreroll\n");
    return GST_FLOW_OK;
}

// onBuffer
GstFlowReturn gstVideo::onBuffer(_GstAppSink* sink, void* user_data) {
    if(!user_data) {
        return GST_FLOW_OK;
    }
    
    gstVideo* dec = (gstVideo*) user_data;
    
    dec->checkBuffer();
    dec->checkMsgBus();
    
    return GST_FLOW_OK;
}

// Capture
bool gstVideo::Capture(void** cpu, void** cuda, unsigned long timeout) {
    mWaitMutex->lock();
    const bool wait_result = mWaitEvent->wait(mWaitMutex, timeout);
    mWaitMutex->unlock();
    
    if(!wait_result) {
        return false;
    }
    
    mRingMutex->lock();
    const uint32_t latest = mLatestRingbuffer;
    const bool received = mLatestReceived;
    mLatestReceived = true;
    mRingMutex->unlock();
    
    // skip if it was already received
    if(received)
        return false;
        
    if(cpu != NULL)
        *cpu = mRingBufferCPU[latest];
    
    if(cuda != NULL)
        *cuda = mRingBufferGPU[latest];
        
    return true;
}

#define release_return { gst_sample_unref(gstSample); return; }

// checkBuffer
void gstVideo::checkBuffer() {
    if(!mAppSink)
        return;
        
    // block waiting for the buffer
    GstSample* gstSample = gst_app_sink_pull_sample(mAppSink);
    
    if(!gstSample) {
        printf(LOG_GSTREAMER "gstreamer video -- gst_app_sink_pull_sample()");
        printf(" returned NULL...\n");
        return;
    }
    
    GstBuffer* gstBuffer = gst_sample_get_buffer(gstSample);
    
    if(!gstBuffer) {
        printf(LOG_GSTREAMER "gstreamer video -- gst_sample_get_buffer()");
        printf(" returned NULL...\n");
        return;
    }
    
    // retrive
    GstMapInfo map;
    
    if(!gst_buffer_map(gstBuffer, &map, GST_MAP_READ)) {
        printf(LOG_GSTREAMER "gstreamer video -- gst_buffer_map() ");
        printf("failed...\n");
        return;
    }
    
    void* gstData = map.data;
    const uint32_t gstSize = map.size;
    
    if(!gstData) {
        printf(LOG_GSTREAMER "gstreamer video -- gst_buffer had NULL ");
        printf("data pointer...\n");
        return;
    }
    
    // retrive caps
    GstCaps* gstCaps = gst_sample_get_caps(gstSample);
    
    if(!gstCaps) {
        printf(LOG_GSTREAMER "gstreamer video -- gst_buffer had NULL ");
        printf("caps...\n");
        return;
    }
    
    GstStructure* gstCapsStruct = gst_caps_get_structure(gstCaps, 0);
    
    if(!gstCapsStruct) {
        printf(LOG_GSTREAMER "gstreamer video -- gst_caps had NULL ");
        printf("structure...\n");
        return;
    }
    
    // get width and height of the buffer
    int width = 0;
    int height = 0;
    
    if(!gst_structure_get_int(gstCapsStruct, "width", &width) ||
       !gst_structure_get_int(gstCapsStruct, "height", &height)) {
           printf(LOG_GSTREAMER " gstreamer video -- gst_caps missing ");
           printf("width/height...\n");
           return;
    }
    
    if(width < 1 || height < 1)
        release_return;
        
    mWidth = width;
    mHeight = height;
    mDepth = (gstSize * 8)/(width * height);
    mSize = gstSize;
    
    // make sure ringbuffer is allocated
    if(!mRingBufferCPU[0]) {
        for(uint32_t n = 0; n < NUM_RINGBUFFERS; n++) {
            if(!cudaAllocMapped(&mRingBufferCPU[n], &mRingBufferGPU[n], 
                gstSize)){
                printf(LOG_GSTREAMER "gstreamer video -- failed to allocate");
                printf("ringbuffer %u (size=%u)\n", n, gstSize);
            }
        }
        printf(LOG_GSTREAMER "gstreamer video -- allocated %u ", 
                NUM_RINGBUFFERS);
        printf("ringbuffers, %u bytes each\n", gstSize);
    }
    
    //copy to next ringbuffer
    const uint32_t nextRingbuffer = (mLatestRingbuffer + 1) % NUM_RINGBUFFERS;
    mFrameCount++;
    memcpy(mRingBufferCPU[nextRingbuffer], gstData, gstSize);
    gst_buffer_unmap(gstBuffer, &map);
    gst_sample_unref(gstSample);
    
    // update and signal sleeping threads
	mRingMutex->lock();
	mLatestRingbuffer = nextRingbuffer;
	mLatestReceived  = false;
	mRingMutex->unlock();
	mWaitEvent->wakeAll();
}

// Create with all the parameters
gstVideo* gstVideo::Create(const char* videoPath, 
                           uint32_t width,
                           uint32_t height, 
                           uint32_t depth,
                           bool downscaled) {
	if (!gstreamerInit()) {
		printf(LOG_GSTREAMER "failed to initialize gstreamer API\n");
		return NULL;
	}
	
	gstVideo* vid = new gstVideo();
	
	if (!vid)
		return NULL;
	
	vid->mWidth      = width;
	vid->mHeight     = height;
	vid->mDepth      = depth; // should be 24 for RGB or 12 for NV12
	vid->mSize       = (width * height * vid->mDepth) / 8;
	
	// pipeline is set here
	std::ostringstream pipeline;
	
    if (downscaled) {
        pipeline << "filesrc location=" << videoPath
                 << " ! decodebin"
                 << " ! nvvidconv" //left=756 top=222 right=2036 bottom=942"
                 << " ! video/x-raw, format=(string)NV12"
                 << ", width=(int)" << width << ", height=(int)" << height
                 << " ! clockoverlay halignment=right valignment=top shaded-background=true shading-value=120 time-format=\"%Y/%m/%d %H:%M:%S\""
                 << " ! appsink name=mysink";
    } else {
        pipeline << "filesrc location=" << videoPath
                 << " ! decodebin"
                 << " ! nvvidconv" //left=756 top=222 right=2036 bottom=942"
                 << " ! video/x-raw, format=(string)I420"
                 << " ! appsink name=mysink";
    }
	
	vid->mLaunchStr = pipeline.str();

	if (!vid->init()) {
		printf(LOG_GSTREAMER "failed to init gstVideo\n");
		return NULL;
	}
	
	return vid;
}

gstVideo* gstVideo::Create(const char* videoPath, 
                           uint32_t width,
                           uint32_t height,
                           bool downscaled) {
	if (!gstreamerInit()) {
		printf(LOG_GSTREAMER "failed to initialize gstreamer API\n");
		return NULL;
	}
	
	gstVideo* vid = new gstVideo();
	
	if (!vid)
		return NULL;
	
	vid->mWidth      = width;
	vid->mHeight     = height;
	vid->mDepth      = DefaultDepth; // should be 24 for RGB or 12 for NV12
	vid->mSize       = (width * height * vid->mDepth) / 8;
	
	// pipeline is set here
	std::ostringstream pipeline;
	
    if (downscaled) {
        pipeline << "filesrc location=" << videoPath
                 << " ! decodebin"
                 << " ! nvvidconv"// left=756 top=222 right=2036 bottom=942"
                 << " ! video/x-raw, format=(string)NV12"
                 << ", width=(int)" << width << ", height=(int)" << height
                 << " ! clockoverlay halignment=right valignment=top shaded-background=true shading-value=120 time-format=\"%Y/%m/%d %H:%M:%S\""
                 << " ! appsink name=mysink";
    } else {
        pipeline << "filesrc location=" << videoPath
                 << " ! decodebin"
                 << " ! nvvidconv"// left=756 top=222 right=2036 bottom=942"
                 << " ! video/x-raw, format=(string)I420"
                 << " ! appsink name=mysink";
    }
    printf(LOG_GSTREAMER "Video pipeline: %s \n", pipeline.str().c_str());

	vid->mLaunchStr = pipeline.str();

	if (!vid->init()) {
		printf(LOG_GSTREAMER "failed to init gstVideo\n");
		return NULL;
	}
	
	return vid;
}

// Create with just video name (path)
gstVideo* gstVideo::Create(const char* videoPath, bool downscaled) {
    return Create(videoPath, DefaultWidth, DefaultHeight, DefaultDepth,
                  downscaled);
}


// initialize
bool gstVideo::init() {
    GError* err = NULL;
    
    // launch pipeline
    mPipeline = gst_parse_launch(mLaunchStr.c_str(), &err);
    
    if( err != NULL ) {
		printf(LOG_GSTREAMER "gstreamer decoder failed to create pipeline\n");
		printf(LOG_GSTREAMER "   (%s)\n", err->message);
		g_error_free(err);
		return false;
	}

	GstPipeline* pipeline = GST_PIPELINE(mPipeline);

	if( !pipeline ) {
		printf(LOG_GSTREAMER "gstreamer failed to cast GstElement into ");
		printf("GstPipeline\n");
		return false;
	}

	// retrieve pipeline bus
	/*GstBus**/ mBus = gst_pipeline_get_bus(pipeline);

	if( !mBus ) {
		printf(LOG_GSTREAMER "gstreamer failed to retrieve GstBus from ");
		printf("pipeline\n");
		return false;
	}
	
	// get the appsrc
	GstElement* appsinkElement = gst_bin_get_by_name(GST_BIN(pipeline), 
	                                "mysink");
	GstAppSink* appsink = GST_APP_SINK(appsinkElement);

	if( !appsinkElement || !appsink)
	{
		printf(LOG_GSTREAMER "gstreamer failed to retrieve AppSink element ");
		printf("from pipeline\n");
		return false;
	}
	
	mAppSink = appsink;
	
	// setup callbacks
	GstAppSinkCallbacks cb;
	memset(&cb, 0, sizeof(GstAppSinkCallbacks));
	
	cb.eos         = onEOS;
	cb.new_preroll = onPreroll;
	cb.new_sample  = onBuffer;
	
	gst_app_sink_set_callbacks(mAppSink, &cb, (void*)this, NULL);
	
	return true;
}

// Open
bool gstVideo::Open() {
	// transition pipline to STATE_PLAYING
	printf(LOG_GSTREAMER "gstreamer transitioning pipeline to ");
	printf("GST_STATE_PLAYING\n");
	
	const GstStateChangeReturn result = gst_element_set_state(mPipeline, 
	                                    GST_STATE_PLAYING);

	if( result == GST_STATE_CHANGE_ASYNC ) {
#if 0
		GstMessage* asyncMsg = gst_bus_timed_pop_filtered(mBus, 5 * GST_SECOND, 
    	 					      (GstMessageType)(GST_MESSAGE_ASYNC_DONE|
    	 					      GST_MESSAGE_ERROR)); 

		if( asyncMsg != NULL )
		{
			gst_message_print(mBus, asyncMsg, this);
			gst_message_unref(asyncMsg);
		}
		else
			printf(LOG_GSTREAMER "gstreamer NULL message after ");
			printf("transitioning pipeline to PLAYING...\n");
#endif
	}
	else if( result != GST_STATE_CHANGE_SUCCESS ) {
		printf(LOG_GSTREAMER "gstreamer failed to set pipeline state to ");
		printf("PLAYING (error %u)\n", result);
		return false;
	}

	checkMsgBus();
	usleep(100*1000);
	checkMsgBus();

	return true;
}
	

// Close
void gstVideo::Close() {
	// stop pipeline
	printf(LOG_GSTREAMER "gstreamer transitioning pipeline to ");
	printf("GST_STATE_NULL\n");

	const GstStateChangeReturn result = gst_element_set_state(mPipeline, 
	                                    GST_STATE_NULL);

	if( result != GST_STATE_CHANGE_SUCCESS )
		printf(LOG_GSTREAMER "gstreamer failed to set pipeline state to ");
		printf("PLAYING (error %u)\n", result);

	usleep(250*1000);
}


// checkMsgBus
void gstVideo::checkMsgBus() {
	while (true) {
		GstMessage* msg = gst_bus_pop(mBus);

		if (!msg)
			break;

		gst_message_print(mBus, msg, this);
		gst_message_unref(msg);
	}
}
