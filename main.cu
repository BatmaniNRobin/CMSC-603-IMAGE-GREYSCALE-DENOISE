#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <tuple>
// #include <opencv4/opencv.hpp>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define THREADS_DIM 32
#define WINDOW_SIZE (3)


__global__ void greyscale(uchar4* d_rgb, uchar* d_grey, int matrixHeight, int matrixWidth, int numPixels)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < matrixWidth && row < matrixHeight)
    {
        int rgb_ab = row * matrixWidth + col;
	    uchar4 rgb_image = d_rgb[rgb_ab];
        double gray_val = (float(rgb_image.x))*0.299f + (float(rgb_image.y))*0.587f + (float(rgb_image.z))*0.114f;
        d_grey[rgb_ab] = (unsigned char)gray_val;
    }
}

__global__ void denoise(uchar *d_grey, uchar *d_output, int matrixHeight, int matrixWidth, int numPixels)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned char array[9];

    if(col < matrixWidth && row < matrixHeight)
    {
        for(int x = 0; x < WINDOW_SIZE; x++)
        {
            for(int y = 0; y < WINDOW_SIZE; y++)
            {
                array[x*WINDOW_SIZE+y] = d_grey[(row+x-1)*matrixWidth+(col+y-1)];
            }
        }

        // TODO insertion sort
        
        int rgb_ab = row * matrixWidth + col;
        
        // int i, key, j;

        // __syncthreads();
        // // insertion sort
        // for(i = 1; i < 9; i++)
        // {
        //     key = array[i];
        //     j = i - 1;

        //     while(j >=0 && array[j] > key)
        //     {
        //         array[j+1] = array[j];
        //         j = j - 1;
        //     }
        //     array[j+1] = key;
        // }

        // // write value to d_output
        // d_output[rgb_ab] = (unsigned char) array[4];

        // bubblesort works ...
        for (int i = 0; i < 9; i++) {
            for (int j = i + 1; j < 9; j++) {
                if (array[i] > array[j]) { 
                    //Swap the variables.
                    unsigned char temp = array[i];
                    array[i] = array[j];
                    array[j] = temp;
                }
            }
        }
        d_output[rgb_ab] = (unsigned char) array[4];
    }
}


int main(int argc, char *argv[])
{
    if(argc != 1)
    {
        cout << "Usage: ./main" << endl;
        exit(0);
    }

    Mat img_RGB;
    Mat img_Grey;

    // load image into matrix obj in BGR
    Mat image = imread("image.jpg", CV_LOAD_IMAGE_COLOR);

    // cvtColor(image, BGR2RGB);

    // Check for failure
    if (image.empty()) 
    {
        cout << "Could not open or find the image" << endl;
        exit(1);
    }

    // convert color from openCV standard BGR to RGB
    cvtColor(image, img_RGB, CV_BGR2RGBA);

    // imshow("Image", image); // lol how do i know if my images are greyscale

    // // allocate memory for an imaeg to be saved that is the greyscale version that it should get saved to
    // single channel 8bit color ie grey and with uchar instead of uchar4
    img_Grey.create(image.rows, image.cols, CV_8UC1);

    // struct timespec start, end;
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // // allocate matrices on host and device
    uchar4 *d_rgb, *h_rgb;
    uchar *d_grey;

    h_rgb = (uchar4*)img_RGB.ptr<uchar>(0);

    int matrixWidth = image.cols;
    int matrixHeight = image.rows;

    int numPixels = matrixWidth * matrixHeight;

    cudaMalloc(&d_rgb, numPixels * sizeof(uchar4));
    cudaMalloc(&d_grey, numPixels * sizeof(uchar));

    // // transfer from host to device
    cudaMemcpy(d_rgb, h_rgb, numPixels * sizeof(uchar4), cudaMemcpyHostToDevice);

    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess) 
	{
        const char * errorMessage = cudaGetErrorString(code);
        printf("cuda error HtoD %s", errorMessage);
    }

    // TODO check gridDim
    // call kernel to compute matrix multiplication 
    int gridDim_x = (image.cols / THREADS_DIM) + 1;
    int gridDim_y = (image.rows / THREADS_DIM) + 1;

    dim3 blockSize (THREADS_DIM, THREADS_DIM);
    dim3 gridSize (gridDim_x, gridDim_y);

    // clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    cudaEventRecord(start);

    greyscale <<< gridSize, blockSize >>> (d_rgb, d_grey, matrixHeight, matrixWidth, numPixels);

    // clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    // // transfer back from device to host
    cudaMemcpy(img_Grey.ptr<uchar>(0), d_grey, numPixels * sizeof(uchar), cudaMemcpyDeviceToHost);

    code = cudaGetLastError();
    if (code != cudaSuccess) 
	{
        const char * errorMessage = cudaGetErrorString(code);
        printf("\ncuda error DtoH %s\n\n", errorMessage);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    imwrite("grey.jpg", img_Grey);

    printf("GPU Time for Grey: %f ms\n", milliseconds);

    //------------------------------------------------ DENOISE ---------------------------------------------------

    Mat denoised_image;

    float ms;
    cudaEvent_t begin, finish;
    cudaEventCreate(&begin);
    cudaEventCreate(&finish);

    denoised_image.create(image.rows, image.cols, CV_8UC1);

    uchar *d_output;

    cudaMalloc(&d_output, numPixels * sizeof(uchar));
    
    cudaEventRecord(begin);

    denoise <<< gridSize, blockSize >>> (d_grey, d_output, matrixHeight, matrixWidth, numPixels);

    cudaMemcpy(denoised_image.ptr<uchar>(0), d_output, numPixels * sizeof(uchar), cudaMemcpyDeviceToHost);

    cudaEventRecord(finish);
    cudaEventSynchronize(finish);
    cudaEventElapsedTime(&ms, begin, finish);

    imwrite("denoise.jpg", denoised_image);

    printf("GPU Time for Denoising: %f ms\n", ms);

    // // free memory
    cudaFree(d_rgb);
    cudaFree(d_grey);
    cudaFree(d_output);
}