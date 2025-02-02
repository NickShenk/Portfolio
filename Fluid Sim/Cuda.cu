#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <windows.h>
#include <time.h>
#include <math.h>

// CUDA Kernel: Populates a grid with RGB color values
__global__ void colorGrid(COLORREF* screen, int width, int height, double time, int simwid, int simheight, float* masses) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < width * height) {
        int idx = threadId % width;
        int idy = threadId / width;
        int pixelIndex = idy * width + idx;
        int simx = ((float)idx / width)*simwid;
        int simy = ((float)idy / height)*simheight;
        int intensity;
        if (simx >= 0 && simx < simwid && simy >= 0 && simy < simheight) {
            int massIndex = simy * simwid + simx;
            intensity = max(0, min(255, (int)(masses[massIndex] * 255)));
            // Use intensity in RGB here
        } else {
            intensity = 0; // Default intensity for out-of-bounds cases
        }
        screen[pixelIndex] = RGB(intensity, intensity / 2, 0);
    }
}
// Function to launch the CUDA kernel
static COLORREF* d_screen = nullptr;
static size_t screenSize = 0;
static int currentWidth = 0;
static int currentHeight = 0;
static float* d_masses = nullptr;

extern "C" void LaunchCudaKernel(COLORREF* screen, int width, int height, double time, int simwid, int simheight, float* masses) {
    // Check if dimensions have changed
    if (width != currentWidth || height != currentHeight) {
        // Free the old device memory if dimensions have changed
        if (d_screen) {
            cudaFree(d_screen);
        }

        // Update dimensions and allocate new memory
        currentWidth = width;
        currentHeight = height;
        screenSize = width * height * sizeof(COLORREF);
        cudaMalloc(&d_screen, screenSize);
    }
    if (!d_masses) cudaMalloc(&d_masses, simwid * simheight * sizeof(float));
    cudaMemcpy(d_masses, masses, simwid*simheight*sizeof(float), cudaMemcpyHostToDevice);
    // Define CUDA grid and block dimensions
    dim3 blockDim(256);
    dim3 gridDim((width * height + blockDim.x - 1) / blockDim.x);
    // Launch the kernel
    colorGrid<<<gridDim, blockDim>>>(d_screen, width, height, time, simwid, simheight, d_masses);
    cudaDeviceSynchronize();  // Wait for kernel completion
    // Copy the result back to host memory
    cudaMemcpy(screen, d_screen, screenSize, cudaMemcpyDeviceToHost);
    
}
extern "C" void FreeCudaResources() {
    if (d_screen) {
        cudaFree(d_screen);
        d_screen = nullptr;
    }
    if (d_masses) {
        cudaFree(d_masses);
        d_masses = nullptr;
    }
    currentWidth = 0;
    currentHeight = 0;
}