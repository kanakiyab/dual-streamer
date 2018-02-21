/*
 * This file is inspired by the original work by Nvidia author dusty-nv
 * and the work by omaralvarez for RTSP Pipeline. Their codes are available 
 * on their respective GitHub profiles under the repo jetson-inference.
 * The objective of this file is to make a pipeline that is generic to video 
 * or stream by defining the gst-pipeline as one of the commandline inputs.
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

#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include "commandLine.h"
#include "loadImage.h"

#include "cudaNormalize.h"

bool signal_recieved = false;

void sig_handler(int signo) {
    if(signo == SIGINT) {
        printf("received SIGINT");
        signal_recieved == true;
    }
}

int main(int argc, char** argv) {
    printf("gst-video\n  args (%i):  ", argc);
    
    for (int i=0; i<argc; i++)
        printf("%i [%s]  ", i, argv[i]);
        
    printf("\n");
    
    commandLine cmdLine(argc, argv);

    if (signal(SIGINT, sig_handler) == SIG_ERR)
        printf("\ncant catch SIGINT");
        
    // Create the video pipeline
    
    const char* videoPath = NULL;

    videoPath = cmdLine.GetString("input");

    if (videoPath == NULL) {
        printf("Please input the path to test footage...\n");
        printf("\nUse option --input /path/to/footage.mp4\n");
        return -1;
    }
    
    bool scale = true;
    gstVideo* video = gstVideo::Create(videoPath, scale);

    if (!video) {
        printf("\ngst-video:  failed to load the video\n");
    }

    uint frameWidth = video->GetWidth();
    uint frameHeight = video->GetHeight();

	printf("\ngst-video:  successfully initialized video device\n");
    printf("    width:  %u\n", frameWidth);
    printf("   height:  %u\n", frameHeight);
	printf("    depth:  %u (bpp)\n", video->GetPixelDepth());
	
    // Create openGL window
    glDisplay* display = glDisplay::Create();
	
    if (!display)
		printf("\ngst-video:  failed to create openGL display\n");

    const size_t texSz = frameWidth * frameHeight *
                         sizeof(float4);
    float4* texIn = (float4*)malloc(texSz);
	
    if (texIn != NULL)
        for (uint32_t y=0; y < frameHeight; y++)
            for (uint32_t x=0; x < frameWidth; x++)
                texIn[y*frameWidth+x] = make_float4(0.0f, 1.0f, 1.0f, 1.0f);

    glTexture* texture = glTexture::Create(frameWidth,
                                     frameHeight,
                                     GL_RGBA32F_ARB/*GL_RGBA8*/, texIn);

    if( !texture )
        printf("gst-video:  failed to create openGL texture\n");
		
    // the video resolution for hi res is hard coded for now
    // TODO: Need to fix this
    gstVideo* videoUnscaled = gstVideo::Create(videoPath, false);

    // Start playing
    if (!video->Open()) {
        printf("\ngst-video:  failed to open video for streaming\n");
        return 0;
    }

    if (!videoUnscaled->Open()) {
        printf("\ngst-video:  failed to open high res video for streaming\n");
        return 0;
    }
	
	bool zeroCopy = true;
	printf("\ngst-video:  video open for streaming\n");

	while (!signal_recieved) {
	    void* imgCPU = NULL;
        void* imgCUDA = NULL;
	    
        void* bimgCPU = NULL;
//        void* bimgCUDA = NULL;

        // get the latest frame
        if (!video->Capture(&imgCPU, &imgCUDA, 1000)) {
	        printf("\ngst-video:  failed to capture frame\n");
	        goto cleanup;
        } else {
            printf("gst-video:  received new frame CPU=0x%p, GPU=0x%p\n",
                    imgCPU, imgCUDA);
        }

        // get the latest frame
        if (!videoUnscaled->Capture(&bimgCPU, NULL, 5000)) {
            printf("\ngst-video:  failed to capture hi res frame\n");
            goto cleanup;
        } else {
            printf("gst-video:  received new hi res frame CPU=0x%p\n", bimgCPU);
        }
        
        // convert the NV12 frame to RGBA
        void* imgRGBA = NULL;
        
        if( !video->ConvertRGBA(imgCUDA, &imgRGBA, zeroCopy) )
			printf("gst-video:  failed to convert from NV12 to RGBA\n");

		// rescale image pixel intensities
        CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f),
                               (float4*)imgRGBA, make_float2(0.0f, 1.0f),
                               frameWidth, frameHeight));

        uint32_t frameCount = video->GetFrameCount();
        printf("Frame count: %i\n", frameCount);

 		if (display != NULL) {
			char str[256];
			sprintf(str, "Video streaming | %04.1f FPS", display->GetFPS());
			display->SetTitle(str);	
		}
			
        // update the display
        if (display != NULL) {
            display->UserEvents();
            display->BeginRender();
            
            if (texture != NULL) {
                void* tex_map = texture->MapCUDA();
                
                if (tex_map != NULL) {
                    cudaMemcpy(tex_map, imgRGBA, texture->GetSize(),
                               cudaMemcpyDeviceToDevice);
                               
                    texture -> Unmap();
                }
                
                texture->Render(100, 100);
            }
            display->EndRender();
        }
	}
    
cleanup:
    printf("\ngst-video:  cleaning up video data.\n");
    
    if (video != NULL) {
        delete video;
        video = NULL;
    }
    
    if (display != NULL) {
        delete display;
        display = NULL;
    }
    
    printf("gst-video:  video data has been cleaned.\n");
    printf("gst-video:  this concludes the test of video pipeline.\n");
    return 0;
}
