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

#include "glDisplay.h"
#include "glTexture.h"

#include "commandLine.h"
#include "loadImage.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include "cudaNormalize.h"

#define DEFAULT_CAMERA 1    // rtsp camera

bool signal_recieved = false;

void sig_handler(int signo) {
    if (signo == SIGINT) {
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}


int main(int argc, char** argv) {
	printf("gst-camera\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n");
	
    commandLine cmdLine(argc, argv);

	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");

	/*
	 * create the camera device
	 */
    int width = cmdLine.GetInt("width");
    int height = cmdLine.GetInt("height");
    gstCamera* camera = NULL;

    if (!width || !height) {
        camera = gstCamera::Create(true, DEFAULT_CAMERA);
        width = camera->GetWidth();
        height = camera->GetHeight();
    } else {
        camera = gstCamera::Create(width, height, true, DEFAULT_CAMERA);
    }

	
    if (!camera) {
		printf("\ngst-camera:  failed to initialize video device\n");
		return 0;
	}
	
	printf("\ngst-camera:  successfully initialized video device\n");
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n", camera->GetPixelDepth());

    // the video resolution for hi res is hard coded for now
    // TODO: Need to fix this
    gstCamera* hiResCam = gstCamera::Create(2560, 1440, false,  DEFAULT_CAMERA);

	/*
	 * create openGL window
	 */
    glDisplay* display = NULL;
    glTexture* texture = NULL;

    display = glDisplay::Create();
    if( !display )
        printf("\ngst-camera:  failed to create openGL display\n");

    const size_t texSz = camera->GetWidth() * camera->GetHeight() * sizeof(float4);
    float4* texIn = (float4*)malloc(texSz);

    /*if( texIn != NULL )
        memset(texIn, 0, texSz);*/

    if( texIn != NULL )
        for( uint32_t y=0; y < camera->GetHeight(); y++ )
            for( uint32_t x=0; x < camera->GetWidth(); x++ )
                texIn[y*camera->GetWidth()+x] = make_float4(0.0f, 1.0f, 1.0f, 1.0f);

    texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(),
                                GL_RGBA32F_ARB/*GL_RGBA8*/, texIn);

    if( !texture )
        printf("gst-camera:  failed to create openGL texture\n");

	/*
	 * start streaming
	 */
    if (!camera->Open()) {
		printf("\ngst-camera:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("\ngst-camera:  camera open for streaming\n");
//    const char* fileName = "test-cam.jpg";
//    uint frameCount = 0;
    while (!signal_recieved) {
		void* imgCPU  = NULL;
		void* imgCUDA = NULL;

        void* bimgCPU = NULL;
		
		// get the latest frame
        if (!camera->Capture(&imgCPU, &imgCUDA, 1000))
			printf("\ngst-camera:  failed to capture frame\n");
		else
            printf("gst-camera:  recieved new frame  CPU=0x%p  GPU=0x%p\n",
                   imgCPU, imgCUDA);

        if (!hiResCam->Capture(&bimgCPU, NULL, 5000))
            printf("\ngst-camera: failed to capture hi-res frame\n");
        else
            printf("gst-camera:  recieved new hi res frame CPU=0x%p\n", imgCPU);
		
		// convert from YUV to RGBA
		void* imgRGBA = NULL;
		
        if (camera->ConvertRGBA(imgCUDA, &imgRGBA, true))
			printf("gst-camera:  failed to convert from NV12 to RGBA\n");

        // tracking and reporting code here...
        // ...

		// rescale image pixel intensities
		CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f), 
						   (float4*)imgRGBA, make_float2(0.0f, 1.0f), 
 						   camera->GetWidth(), camera->GetHeight()));

        if (display != NULL) {
            char str[256];
            sprintf(str, "Video streaming | %04.1f FPS", display->GetFPS());
            display->SetTitle(str);
        }

		// update display
        if (display != NULL) {
			display->UserEvents();
			display->BeginRender();

            if (texture != NULL) {
				void* tex_map = texture->MapCUDA();

                if (tex_map != NULL) {
                    cudaMemcpy(tex_map, imgRGBA, texture->GetSize(),
                               cudaMemcpyDeviceToDevice);
					CUDA(cudaDeviceSynchronize());

					texture->Unmap();
				}
				//texture->UploadCPU(texIn);

				texture->Render(100,100);		
			}

			display->EndRender();
		}
	}
	
	printf("\ngst-camera:  un-initializing video device\n");
	
	/*
	 * shutdown the camera device
	 */
    if (camera != NULL) {
		delete camera;
		camera = NULL;
	}

    if (display != NULL) {
		delete display;
		display = NULL;
	}
	
	printf("gst-camera:  video device has been un-initialized.\n");
	printf("gst-camera:  this concludes the test of the video device.\n");
	return 0;
}
