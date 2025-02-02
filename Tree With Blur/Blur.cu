#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <windows.h>
#include <time.h>
#include <cmath> 
#define blurradius 10

__device__ int blursize = blurradius;
// CUDA Kernel: Populates a grid with RGB color values
__device__ int glow(int color) {
    // return 255; // add box to show pixels being drawn
    double cutoff = 70.0;
    double cutoffExtreme = 20.0;
    if (color < cutoff) return ((cutoff + cutoffExtreme)*color) / (color + cutoffExtreme); // make dark colors 25% brighter
    return color;
}
__device__ double decayfunc(double dist) {
    double decay = 1.3;
    return 1.0/((dist+1)*decay);
}
__global__ void blurh(COLORREF* oldscreen, COLORREF* screen, int width, int height, int minx, int miny, int maxx, int maxy) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < (maxx - minx) * (maxy - miny)) {
        double sumR = 0;
        double sumG = 0;
        double sumB = 0;
        double pixelsused = 0;
        int idx = (threadId % (maxx - minx)) + minx;
        int idy = (threadId / (maxx - minx)) + miny;
        int pixelIndex = (idy) * width + (idx);
        
        // calculate horizontal
        for (int x = -blursize; x <= blursize; x++) {
            if (idx + x < width && idx + x > -1) {
                    int oldpixelIndex = (idy) * width + (idx + x);
                    // Calculate the distance from the center pixel
                    double distance = abs(x);
                    // Decay the intensity with distance (apply a glow fade with distance)
                    double distanceDecay = decayfunc(distance);
                    sumR += distanceDecay*glow(GetRValue(oldscreen[oldpixelIndex]));
                    sumB += distanceDecay*glow(GetBValue(oldscreen[oldpixelIndex]));
                    sumG += distanceDecay*glow(GetGValue(oldscreen[oldpixelIndex]));
                    pixelsused += distanceDecay;
            }
        }
        screen[pixelIndex] = RGB(
            min(255,(int) (sumR / pixelsused))
            ,
            min(255,(int) (sumG / pixelsused))
            ,
            min(255,(int) (sumB / pixelsused))
            );
        
    }
}
__global__ void blurv(COLORREF* oldscreen, COLORREF* screen, int width, int height, int minx, int miny, int maxx, int maxy) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < (maxx - minx) * (maxy - miny)) {
        int idx = (threadId % (maxx - minx)) + minx;
        int idy = (threadId / (maxx - minx)) + miny;
        int pixelIndex = (idy) * width + (idx);
        // double intensity = 1.6;
        // double cutoff = 150.0;
         // Decay factor for distance from center
        // calculate vertical
        double sumR = 0;
        double sumG = 0;
        double sumB = 0;
        double pixelsused = 0;
        for (int y = -blursize; y <= blursize; y++) {
            if (idy + y < height && idy + y > -1) {
                int oldpixelIndex = (idy + y) * width + (idx);
                // Calculate the distance from the center pixel
                double distance = abs(y);
                // Decay the intensity with distance (apply a glow fade with distance)
                double distanceDecay = decayfunc(distance);
                sumR += (int) (distanceDecay*(GetRValue(screen[oldpixelIndex])));
                sumB += (int) (distanceDecay*(GetBValue(screen[oldpixelIndex])));
                sumG += (int) (distanceDecay*(GetGValue(screen[oldpixelIndex])));
                pixelsused += distanceDecay;
            }
        }
        oldscreen[pixelIndex] = RGB(
            min(255,(int) (sumR / pixelsused))
            ,
            min(255,(int) (sumG / pixelsused))
            ,
            min(255,(int) (sumB / pixelsused))
            );
        //screen[pixelIndex] = RGB(0,0,0);
    }
}
// Function to launch the CUDA kernel
static COLORREF* d_screen = nullptr;
static COLORREF* d_screenOLD = nullptr;
static size_t screenSize = 0;
static int currentWidth = 0;
static int currentHeight = 0;

extern "C" void LaunchCudaKernel(COLORREF* oldscreen, COLORREF* screen, int width, int height, int minx, int miny, int maxx, int maxy) {
    // Check if dimensions have changed
    if (width != currentWidth || height != currentHeight) {
        // Free the old device memory if dimensions have changed
        if (d_screen) {
            cudaFree(d_screen);
            cudaFree(d_screenOLD);
            
            
        }

        // Update dimensions and allocate new memory
        currentWidth = width;
        currentHeight = height;
        screenSize = width * height * sizeof(COLORREF);
        cudaMalloc(&d_screen, screenSize);
        cudaMalloc(&d_screenOLD, screenSize);
    }
    
    // Copy the input screen data to the device
    cudaMemcpy(d_screenOLD, oldscreen, screenSize, cudaMemcpyHostToDevice);
    minx -= blurradius;
    minx = max(0,minx);
    miny -= blurradius;
    miny = max(0,miny);
    maxx += blurradius;
    maxx = min(width,maxx);
    maxy += blurradius;
    maxy = min(height, maxy);
    // Define CUDA grid and block dimensions
    dim3 blockDim(1024);
    dim3 gridDim(((maxx - minx) * (maxy-miny) + blockDim.x - 1) / blockDim.x);

    // Launch the kernel
    blurh<<<gridDim, blockDim>>>(d_screenOLD, d_screen, width, height, minx, miny, maxx, maxy);
    blurv<<<gridDim, blockDim>>>(d_screenOLD, d_screen, width, height, minx, miny, maxx, maxy);
    // Copy the result back to host memory
    cudaMemcpy(screen, d_screenOLD, screenSize, cudaMemcpyDeviceToHost); // copy old because it has correct data
}

// Cleanup function to release memory when the application exits
extern "C" void FreeCudaResources() {
    if (d_screen) {
        cudaFree(d_screen);
        d_screen = nullptr;
    }
    if (d_screenOLD) {
        cudaFree(d_screenOLD);
        d_screenOLD = nullptr;
    }
    currentWidth = 0;
    currentHeight = 0;
}
