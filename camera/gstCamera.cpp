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

#include "gstCamera.h"
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


// constructor
gstCamera::gstCamera() {
    mAppSinkLR    = NULL;
    mAppSinkHR    = NULL;
	mBus        = NULL;
	mPipeline   = NULL;	
	mV4L2Device = -1;
	
	mWidth  = 0;
	mHeight = 0;
	mDepth  = 0;
	mSize   = 0;
    mFrameCount = 0;
	
	mWaitEvent  = new QWaitCondition();
	mWaitMutex  = new QMutex();
	mRingMutex  = new QMutex();
	
	mLatestRGBA       = 0;
	mLatestRingbuffer = 0;
	mLatestRetrieved  = false;
	
    for( uint32_t n=0; n < NUM_RINGBUFFERS; n++ ) {
        mRingbufferCPU[n]   = NULL;
        mRingbufferGPU[n]   = NULL;
        mRingbufferHRCPU[n] = NULL;
        mRGBA[n]            = NULL;
	}
}


// destructor	
gstCamera::~gstCamera() {
	
}


// ConvertRGBA
bool gstCamera::ConvertRGBA( void* input, void** output, bool zeroCopy ) {
	if( !input || !output )
		return false;
	
    if( !mRGBA[0] ) {
		const size_t size = mWidth * mHeight * sizeof(float4);

        for( uint32_t n=0; n < NUM_RINGBUFFERS; n++ ) {
            if( zeroCopy ) {
				void* cpuPtr = NULL;
				void* gpuPtr = NULL;

                if( !cudaAllocMapped(&cpuPtr, &gpuPtr, size) ) {
                    printf(LOG_CUDA "gstCamera -- failed to allocate ");
                    printf("zeroCopy memory for %ux%xu RGBA texture\n",
                           mWidth, mHeight);
					return false;
				}

                if( cpuPtr != gpuPtr ) {
                    printf(LOG_CUDA "gstCamera -- zeroCopy memory has diff");
                    printf("erent pointers, please use a UVA-compatible GPU\n");
					return false;
				}

				mRGBA[n] = gpuPtr;
			}
            else {
                if( CUDA_FAILED(cudaMalloc(&mRGBA[n], size)) ) {
                    printf(LOG_CUDA "gstCamera -- failed to allocate ");
                    printf("memory for %ux%u RGBA texture\n", mWidth, mHeight);
					return false;
				}
			}
		}
		
        printf(LOG_CUDA "gstreamer camera -- allocated %u RGBA ringbuffers\n",
               NUM_RINGBUFFERS);
	}
	
    // onboard camera or rtsp is NV12
    if(CUDA_FAILED(cudaNV12ToRGBAf((uint8_t*)input, (float4*)mRGBA[mLatestRGBA],
                                    mWidth, mHeight)))
        return false;
	
	*output     = mRGBA[mLatestRGBA];
    mLatestRGBA = (mLatestRGBA + 1) % NUM_RINGBUFFERS;
	return true;
}


// onEOS
void gstCamera::onEOS(_GstAppSink* sink, void* user_data) {
	printf(LOG_GSTREAMER "gstreamer decoder onEOS\n");
}


// onPreroll
GstFlowReturn gstCamera::onPreroll(_GstAppSink* sink, void* user_data) {
	printf(LOG_GSTREAMER "gstreamer decoder onPreroll\n");
	return GST_FLOW_OK;
}


// onBuffer
GstFlowReturn gstCamera::onBufferLR(_GstAppSink* sink, void* user_data) {
	//printf(LOG_GSTREAMER "gstreamer decoder onBuffer\n");
	
	if( !user_data )
        return GST_FLOW_ERROR;
		
	gstCamera* dec = (gstCamera*)user_data;
	
    dec->checkBufferLR();
	dec->checkMsgBus();
	return GST_FLOW_OK;
}

GstFlowReturn gstCamera::onBufferHR(_GstAppSink* sink, void* user_data) {
    //printf(LOG_GSTREAMER "gstreamer decoder onBuffer\n");

    if( !user_data )
        return GST_FLOW_ERROR;

    gstCamera* dec = (gstCamera*)user_data;

    dec->checkBufferHR();
    dec->checkMsgBus();
    return GST_FLOW_OK;
}

// Capture
bool gstCamera::Capture(void** cpuLR, void** cudaLR, void** cpuHR,
                        unsigned long timeout) {
	mWaitMutex->lock();
    const bool wait_result = mWaitEvent->wait(mWaitMutex, timeout);
    mWaitMutex->unlock();
	
    if (!wait_result) {
        printf(LOG_GSTREAMER "gstreamer-camera: wait timedout...\n");
		return false;
    }
	
	mRingMutex->lock();
	const uint32_t latest = mLatestRingbuffer;
	const bool retrieved = mLatestRetrieved;
	mLatestRetrieved = true;
	mRingMutex->unlock();
	
	// skip if it was already retrieved
    if (retrieved) {
        printf(LOG_GSTREAMER "gstreamer-camera: frame already received!!!\n");
		return false;
    }
	
    if (cpuLR != NULL)
        *cpuLR = mRingbufferCPU[latest];
	
    if (cudaLR != NULL)
        *cudaLR = mRingbufferGPU[latest];
	
    if (cpuHR != NULL)
        *cpuHR = mRingbufferHRCPU[latest];

	return true;
}


#define release_return { gst_sample_unref(gstSample); return; }


// checkBuffer
void gstCamera::checkBufferLR() {
    if (!mAppSinkLR)
		return;

	// block waiting for the buffer
    GstSample* gstSample = gst_app_sink_pull_sample(mAppSinkLR);
	
    if (!gstSample) {
        printf(LOG_GSTREAMER "gstreamer camera -- gst_app_sink_pull_sample() ");
        printf("returned NULL...\n");
		return;
	}
	
	GstBuffer* gstBuffer = gst_sample_get_buffer(gstSample);
	
    if (!gstBuffer) {
        printf(LOG_GSTREAMER "gstreamer camera -- ");
               printf("gst_sample_get_buffer() returned NULL...\n");
		return;
	}
	
	// retrieve
	GstMapInfo map; 

    if (!gst_buffer_map(gstBuffer, &map, GST_MAP_READ)) {
        printf(LOG_GSTREAMER "gstreamer camera -- ");
        printf("gst_buffer_map() failed...\n");
		return;
	}
	
	//gst_util_dump_mem(map.data, map.size); 

	void* gstData = map.data; //GST_BUFFER_DATA(gstBuffer);
	const uint32_t gstSize = map.size; //GST_BUFFER_SIZE(gstBuffer);
	
    if (!gstData) {
        printf(LOG_GSTREAMER "gstreamer camera -- ");
        printf("gst_buffer had NULL data pointer...\n");
		release_return;
	}
	
	// retrieve caps
	GstCaps* gstCaps = gst_sample_get_caps(gstSample);
	
    if (!gstCaps) {
        printf(LOG_GSTREAMER "gstreamer camera -- ");
        printf("gst_buffer had NULL caps...\n");
		release_return;
	}
	
	GstStructure* gstCapsStruct = gst_caps_get_structure(gstCaps, 0);
	
    if (!gstCapsStruct) {
        printf(LOG_GSTREAMER "gstreamer camera -- ");
        printf("gst_caps had NULL structure...\n");
		release_return;
	}
	
	// get width & height of the buffer
	int width  = 0;
	int height = 0;
	
    if (!gst_structure_get_int(gstCapsStruct, "width", &width) ||
        !gst_structure_get_int(gstCapsStruct, "height", &height)) {
        printf(LOG_GSTREAMER "gstreamer camera -- ");
        printf("gst_caps missing width/height...\n");
		release_return;
	}
	
    if (width < 1 || height < 1)
		release_return;
	
	mWidth  = width;
	mHeight = height;
	mDepth  = (gstSize * 8) / (width * height);
	mSize   = gstSize;
	
	//printf(LOG_GSTREAMER "gstreamer camera recieved %ix%i frame (%u bytes, %u bpp)\n", width, height, gstSize, mDepth);
	
	// make sure ringbuffer is allocated
    if (!mRingbufferCPU[0]) {
        for (int32_t n=0; n < NUM_RINGBUFFERS; n++) {
            if (!cudaAllocMapped(&mRingbufferCPU[n], &mRingbufferGPU[n],
                                 gstSize)) {
                printf(LOG_CUDA "gstreamer camera -- failed to allocate ");
                printf("ringbuffer %u  (size=%u)\n", n, gstSize);
            }
		}
		
        printf(LOG_CUDA "gstreamer camera -- allocated ");
        printf("%u ringbuffers, %u bytes each\n", NUM_RINGBUFFERS, gstSize);
	}
	
	// copy to next ringbuffer
    const uint32_t nextRingbuffer = (mLatestRingbuffer + 1) % NUM_RINGBUFFERS;
    // increment framecount
    mFrameCount++;
	//printf(LOG_GSTREAMER "gstreamer camera -- using ringbuffer #%u for next frame\n", nextRingbuffer);
    memcpy(mRingbufferCPU[nextRingbuffer], gstData, gstSize);
	gst_buffer_unmap(gstBuffer, &map); 
	//gst_buffer_unref(gstBuffer);
	gst_sample_unref(gstSample);
	
	
	// update and signal sleeping threads
	mRingMutex->lock();
	mLatestRingbuffer = nextRingbuffer;
	mLatestRetrieved  = false;
	mRingMutex->unlock();
	mWaitEvent->wakeAll();
}

// checkBuffer
void gstCamera::checkBufferHR() {
    if (!mAppSinkHR)
        return;

    // block waiting for the buffer
    GstSample* gstSample = gst_app_sink_pull_sample(mAppSinkHR);

    if (!gstSample) {
        printf(LOG_GSTREAMER "gstreamer camera -- gst_app_sink_pull_sample() ");
        printf("returned NULL...\n");
        return;
    }

    GstBuffer* gstBuffer = gst_sample_get_buffer(gstSample);

    if (!gstBuffer) {
        printf(LOG_GSTREAMER "gstreamer camera -- ");
               printf("gst_sample_get_buffer() returned NULL...\n");
        return;
    }

    // retrieve
    GstMapInfo map;

    if (!gst_buffer_map(gstBuffer, &map, GST_MAP_READ)) {
        printf(LOG_GSTREAMER "gstreamer camera -- ");
        printf("gst_buffer_map() failed...\n");
        return;
    }

    //gst_util_dump_mem(map.data, map.size);

    void* gstData = map.data; //GST_BUFFER_DATA(gstBuffer);
    const uint32_t gstSize = map.size; //GST_BUFFER_SIZE(gstBuffer);

    if (!gstData) {
        printf(LOG_GSTREAMER "gstreamer camera -- ");
        printf("gst_buffer had NULL data pointer...\n");
        release_return;
    }

    // retrieve caps
    GstCaps* gstCaps = gst_sample_get_caps(gstSample);

    if (!gstCaps) {
        printf(LOG_GSTREAMER "gstreamer camera -- ");
        printf("gst_buffer had NULL caps...\n");
        release_return;
    }

    GstStructure* gstCapsStruct = gst_caps_get_structure(gstCaps, 0);

    if (!gstCapsStruct) {
        printf(LOG_GSTREAMER "gstreamer camera -- ");
        printf("gst_caps had NULL structure...\n");
        release_return;
    }

    // get width & height of the buffer
    int width  = 0;
    int height = 0;

    if (!gst_structure_get_int(gstCapsStruct, "width", &width) ||
        !gst_structure_get_int(gstCapsStruct, "height", &height)) {
        printf(LOG_GSTREAMER "gstreamer camera -- ");
        printf("gst_caps missing width/height...\n");
        release_return;
    }

    if (width < 1 || height < 1)
        release_return;

//	mWidth  = width;
//	mHeight = height;
//	mDepth  = (gstSize * 8) / (width * height);
//	mSize   = gstSize;

    //printf(LOG_GSTREAMER "gstreamer camera recieved %ix%i frame (%u bytes, %u bpp)\n", width, height, gstSize, mDepth);

    // make sure ringbuffer is allocated
    if (!mRingbufferCPU[0]) {
        for (int32_t n=0; n < NUM_RINGBUFFERS; n++) {
            mRingbufferHRCPU[n] = (uchar*) malloc(gstSize);
//            if (!cudaAllocMapped(&mRingbufferCPU[n], &mRingbufferGPU[n],
//                                 gstSize)) {
//                printf(LOG_CUDA "gstreamer camera -- failed to allocate ");
//                printf("ringbuffer %u  (size=%u)\n", n, gstSize);
//            }
        }

        printf(LOG_CUDA "gstreamer camera -- allocated ");
        printf("%u ringbuffers, %u bytes each\n", NUM_RINGBUFFERS, gstSize);
    }

    // copy to next ringbuffer
    const uint32_t nextRingbuffer = (mLatestRingbuffer + 1) % NUM_RINGBUFFERS;
    // increment framecount
    mFrameCount++;
    //printf(LOG_GSTREAMER "gstreamer camera -- using ringbuffer #%u for next frame\n", nextRingbuffer);
    memcpy(mRingbufferHRCPU[nextRingbuffer], gstData, gstSize);
    gst_buffer_unmap(gstBuffer, &map);
    //gst_buffer_unref(gstBuffer);
    gst_sample_unref(gstSample);


    // update and signal sleeping threads
    mRingMutex->lock();
    mLatestRingbuffer = nextRingbuffer;
    mLatestRetrieved  = false;
    mRingMutex->unlock();
    mWaitEvent->wakeAll();
}

// buildLaunchStr
bool gstCamera::buildLaunchStr(bool downscaled) {
    // gst-launch-1.0 nvcamerasrc fpsRange="30.0 30.0" ! \
    // 'video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, \
    // format=(string)I420, framerate=(fraction)30/1' ! nvvidconv flip-method=2\
    // ! 'video/x-raw(memory:NVMM), format=(string)I420' ! fakesink silent=false -v
	std::ostringstream ss;
	
    if (onboardCamera()) {
	#if NV_TENSORRT_MAJOR > 1	// if JetPack 3.1 (different flip-method)
		const int flipMethod = 0;
	#else
		const int flipMethod = 2;
	#endif
	
        ss << "nvcamerasrc fpsRange=\"30.0 30.0\"";
        ss << " ! video/x-raw(memory:NVMM), format=(string)NV12,";
        ss << " width=(int)" << mWidth << ", height=(int)" << mHeight;
        ss << " ! nvvidconv flip-method=" << flipMethod ;
        ss << " ! video/x-raw ! appsink name=mysink";
        //'video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,
        // format=(string)I420, framerate=(fraction)30/1' ! ";
    } else {
    // gst-launch-1.0 rtspsrc location=rtsp://admin:Motusing@192.168.1.64/Streaming/Channels/1 latency=100 ! decodebin !
    // tee name=t  t. ! queue ! nvvidconv ! 'video/x-raw, format=I420' ! fakesink t. !
    // queue ! nvvidconv ! 'video/x-raw, width=640, height=360, format=NV12' ! fakesink

        ss << "rtspsrc location=rtsp://admin:Motusing@192.168.1.64/Streaming/Channels/1 latency=100";
        ss << " ! decodebin ! tee name=streams streams. ! queue name=hiresq";
        ss << " ! nvvidconv ! video/x-raw, format=I420";
        ss << " ! appsink name=hiressink streams.";
        ss << " ! queue ! nvvidconv ! video/x-raw, format=NV12,";
        ss << " width=" << mWidth << ", height=" << mHeight;
        ss << " ! clockoverlay halignment=right valignment=top shaded-background=true shading-value=120 time-format=\"%Y/%m/%d %H:%M:%S\"";
        ss << " ! appsink name=loressink";

//        if (downscaled) {
//            ss << "rtspsrc location=rtsp://admin:pass@192.168.1.64/Streaming/Channels/1 latency=100";
//            ss << " ! decodebin";
//            ss << " ! nvvidconv";
//            ss << " ! video/x-raw, format=NV12,";
//            ss << " width=" << mWidth << ", height=" << mHeight;
//            ss << " ! clockoverlay halignment=right valignment=top shaded-background=true shading-value=120 time-format=\"%Y/%m/%d %H:%M:%S\"";
//            ss << " ! appsink name=mysink";
//        } else {
//            ss << "rtspsrc location=rtsp://admin:pass@192.168.1.64/Streaming/Channels/1 latency=100";
//            ss << " ! decodebin";
//            ss << " ! nvvidconv";
//            ss << " ! video/x-raw, format=I420"; /*, width=1920, height=1080";*/
//            ss << " ! appsink name=mysink";
//        }
	}

	mLaunchStr = ss.str();

	printf(LOG_GSTREAMER "gstreamer decoder pipeline string:\n");
	printf("%s\n", mLaunchStr.c_str());
	return true;
}


// Create
gstCamera* gstCamera::Create(uint32_t width, uint32_t height, bool downscaled,
                             int v4l2_device) {
    if (!gstreamerInit()) {
		printf(LOG_GSTREAMER "failed to initialize gstreamer API\n");
		return NULL;
	}
	
	gstCamera* cam = new gstCamera();
	
    if (!cam)
		return NULL;
	
	cam->mV4L2Device = v4l2_device;
	cam->mWidth      = width;
	cam->mHeight     = height;
	cam->mDepth      = cam->onboardCamera() ? 12 : 24;	// NV12 or RGB
	cam->mSize       = (width * height * cam->mDepth) / 8;

    if (!cam->init(downscaled)) {
		printf(LOG_GSTREAMER "failed to init gstCamera\n");
		return NULL;
	}
	
	return cam;
}


// Create
gstCamera* gstCamera::Create(bool downscaled, int v4l2_device) {
    return Create(DefaultWidth, DefaultHeight, downscaled, v4l2_device);
}


// init 
bool gstCamera::init(bool downscaled) {
	GError* err = NULL;

	// build pipeline string
    if (!buildLaunchStr(downscaled)) {
		printf(LOG_GSTREAMER "gstreamer decoder failed to build pipeline string\n");
		return false;
	}

	// launch pipeline
    mPipeline = gst_parse_launch(mLaunchStr.c_str(), &err);

    if (err != NULL) {
		printf(LOG_GSTREAMER "gstreamer decoder failed to create pipeline\n");
		printf(LOG_GSTREAMER "   (%s)\n", err->message);
		g_error_free(err);
		return false;
	}

	GstPipeline* pipeline = GST_PIPELINE(mPipeline);

    if (!pipeline) {
        printf(LOG_GSTREAMER "gstreamer failed to cast GstElement "
                             "into GstPipeline\n");
		return false;
	}	

	// retrieve pipeline bus
	/*GstBus**/ mBus = gst_pipeline_get_bus(pipeline);

    if (!mBus) {
        printf(LOG_GSTREAMER "gstreamer failed to retrieve "
                             "GstBus from pipeline\n");
		return false;
	}

    // add watch for messages (disabled when we poll the bus ourselves,
    // instead of gmainloop)
	//gst_bus_add_watch(mBus, (GstBusFunc)gst_message_print, NULL);

	// get the appsrc
    GstElement* appsinkElementLR = gst_bin_get_by_name(GST_BIN(pipeline), "loressink");
    GstElement* appsinkElementHR = gst_bin_get_by_name(GST_BIN(pipeline), "hiressink");
    GstAppSink* appsinkLR = GST_APP_SINK(appsinkElementLR);
    GstAppSink* appsinkHR = GST_APP_SINK(appsinkElementHR);

    if (!appsinkElementLR || !appsinkLR) {
        printf(LOG_GSTREAMER "gstreamer failed to retrieve lo-res"
                             "AppSink element from pipeline\n");
		return false;
	}

    if (!appsinkElementHR || !appsinkHR) {
        printf(LOG_GSTREAMER "gstreamer failed to retrieve hi-res"
                             "AppSink element from pipeline\n");
        return false;
    }
	
    mAppSinkLR = appsinkLR;
    mAppSinkHR = appsinkHR;
	
	// setup callbacks
    GstAppSinkCallbacks cbLR;
    GstAppSinkCallbacks cbHR;
    memset(&cbLR, 0, sizeof(GstAppSinkCallbacks));
    memset(&cbHR, 0, sizeof(GstAppSinkCallbacks));
	
    cbLR.eos         = onEOS;
    cbLR.new_preroll = onPreroll;
    cbLR.new_sample  = onBufferLR;

    cbHR.eos         = onEOS;
    cbHR.new_preroll = onPreroll;
    cbHR.new_sample  = onBufferHR;
	
    gst_app_sink_set_callbacks(mAppSinkLR, &cbLR, (void*)this, NULL);
    gst_app_sink_set_callbacks(mAppSinkHR, &cbHR, (void*)this, NULL);
	
	return true;
}


// Open
bool gstCamera::Open() {
	// transition pipline to STATE_PLAYING
    printf(LOG_GSTREAMER "gstreamer transitioning pipeline "
                         "to GST_STATE_PLAYING\n");
	
    const GstStateChangeReturn result = gst_element_set_state(mPipeline,
                                                              GST_STATE_PLAYING);

    if (result == GST_STATE_CHANGE_ASYNC) {
#if 0
		GstMessage* asyncMsg = gst_bus_timed_pop_filtered(mBus, 5 * GST_SECOND, 
                                  (GstMessageType)(GST_MESSAGE_ASYNC_DONE|
                                                   GST_MESSAGE_ERROR));

        if (asyncMsg != NULL) {
			gst_message_print(mBus, asyncMsg, this);
			gst_message_unref(asyncMsg);
		}
		else
            printf(LOG_GSTREAMER "gstreamer NULL message after "
                                 "transitioning pipeline to PLAYING...\n");
#endif
	}
    else if (result != GST_STATE_CHANGE_SUCCESS) {
        printf(LOG_GSTREAMER "gstreamer failed to set pipeline state to "
                             "PLAYING (error %u)\n", result);
		return false;
	}

	checkMsgBus();
	usleep(100*1000);
	checkMsgBus();

	return true;
}
	

// Close
void gstCamera::Close() {
	// stop pipeline
	printf(LOG_GSTREAMER "gstreamer transitioning pipeline to GST_STATE_NULL\n");

    const GstStateChangeReturn result = gst_element_set_state(mPipeline,
                                                              GST_STATE_NULL);

    if (result != GST_STATE_CHANGE_SUCCESS) {
        printf(LOG_GSTREAMER "gstreamer failed to set pipeline state "
                             "to PLAYING (error %u)\n", result);
    }

	usleep(250*1000);
}


// checkMsgBus
void gstCamera::checkMsgBus() {
    while(true) {
		GstMessage* msg = gst_bus_pop(mBus);

        if (!msg)
			break;

		gst_message_print(mBus, msg, this);
		gst_message_unref(msg);
	}
}
