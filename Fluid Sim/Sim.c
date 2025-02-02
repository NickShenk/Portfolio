#include <windows.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time()

typedef struct Fluid {
    float mass; // used to decide particle count
    float vx;
    float vy;
} Fluid;


Fluid CreateFluid(float mass, float vx, float vy) {
    Fluid fluid;
    fluid.mass = mass;
    fluid.vx = vx;
    fluid.vy = vy;
    return fluid;
}

// Window procedure function prototype
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
#define SIMWIDTH 200
#define SIMHEIGHT 200
struct Fluid FluidArray[SIMWIDTH*SIMHEIGHT];


LARGE_INTEGER startTime, endTime, frequency;
// CUDA Kernel Declaration
extern void LaunchCudaKernel(COLORREF* screen, int width, int height, double t, int simwid, int simheight, float* masses);
extern  void FreeCudaResources();
// initialize bitmap
static HBITMAP hBitmap = NULL;
static void* pixelData = NULL;
static int height = 1080;
static int width = 1920;

void InitializeDIBSection(HDC hdc, RECT clientRect) {
    if (hBitmap) {
        DeleteObject(hBitmap);
        hBitmap = NULL;
    }
    width = clientRect.right - clientRect.left;
    height = clientRect.bottom - clientRect.top;

    BITMAPINFO bmi = {0};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biHeight = -height; // Negative to make the origin top-left
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32; // 32-bit color
    bmi.bmiHeader.biCompression = BI_RGB;

    hBitmap = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, &pixelData, NULL, 0);
}
float masses[SIMWIDTH*SIMHEIGHT];
float lastMasses[SIMWIDTH*SIMHEIGHT];
// Function to draw using CUDA
void DrawWithCuda(HDC hdc, RECT clientRect, double t) {
    if (!hBitmap || !pixelData || (clientRect.right - clientRect.left != width) || (clientRect.bottom - clientRect.top != height)) {
        InitializeDIBSection(hdc, clientRect);
    }
    for (int m = 0; m < SIMWIDTH*SIMHEIGHT; m++)
        masses[m] = FluidArray[m].mass;
    // Call the CUDA kernel to fill the pixel data
    LaunchCudaKernel((COLORREF*)pixelData, width, height, t, SIMWIDTH, SIMHEIGHT,(float*)masses);

    // Create a memory DC and select the bitmap
    HDC hdcMem = CreateCompatibleDC(hdc);
    HBITMAP hOldBitmap = (HBITMAP)SelectObject(hdcMem, hBitmap);

    // Copy the memory DC to the screen
    BitBlt(hdc, 0, 0, width, height, hdcMem, 0, 0, SRCCOPY);

    // Cleanup
    SelectObject(hdcMem, hOldBitmap);
    DeleteDC(hdcMem);
}
int ind(float x, float y) {
    return round(y)*SIMWIDTH + round(x);
}
int mousedown = FALSE;
double totalt;
// Window procedure function
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    
    switch (uMsg) {
        
        case WM_SIZE:
            InvalidateRect(hwnd, NULL, TRUE);  // Invalidate the window to force a redraw
            break;
        case WM_LBUTTONDOWN : {
            mousedown = TRUE;
            break;
        }
        case WM_LBUTTONUP: {
            mousedown = FALSE;
            break;
        }
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            RECT rect;
            GetClientRect(hwnd, &rect);

            // Measure frame time
            QueryPerformanceCounter(&endTime);

            // make current masses last to reset
            float *temp = lastMasses;
            *lastMasses = *masses;
            *masses = *lastMasses;
            if (mousedown) {
                POINT pt;
                GetCursorPos(&pt);
                ScreenToClient(hwnd, &pt);
                FluidArray[ind((float)(pt.x)/width * SIMWIDTH,(float)pt.y / height * SIMHEIGHT)].mass += 18.0;
            }
            float massExchange = .1;
            for (int i = 0; i < SIMWIDTH; i++) {
                for (int j = 0; j < SIMHEIGHT; j++) {
                    int index = ind(i, j);
                    int randomsign1 = rand() % 2? 1:-1;

                    for (int di = -1; di <= 1; di++) {
                        int randomsign2 = rand() % 2? 1:-1;
                        for (int dj = -1; dj <= 1; dj++) {
                            int neighborIndex = ind(i + di*randomsign1, j + dj*randomsign2);
                            float gravity = 0.0;
                            if (dj*randomsign2 > 0) gravity = .8;
                            if (di == 0 && dj == 0 || (neighborIndex < 0) || (neighborIndex > (SIMWIDTH*SIMHEIGHT - 1))) continue;
                            float diff = FluidArray[index].mass - FluidArray[neighborIndex].mass + gravity*FluidArray[index].mass;
                            if (diff > 0 && FluidArray[neighborIndex].mass < 1.0) {
                                float transfer = diff * (0.5f);
                                FluidArray[index].mass -= transfer;
                                FluidArray[neighborIndex].mass += transfer;
                                break;
                            }
                            // FluidArray[index].mass *= .9999; // decay over time
                        }
                    }
                    
                }
            }

            //for (int m = 0; m < SIMHEIGHT*SIMWIDTH; m++)
                //FluidArray[m].mass = (float)rand() / RAND_MAX;
            double deltaTime;
            QueryPerformanceFrequency(&frequency);
            deltaTime = ((double)(endTime.QuadPart - startTime.QuadPart) / frequency.QuadPart); // in seconds
            totalt += deltaTime;
            QueryPerformanceCounter(&startTime);
            DrawWithCuda(hdc, rect, totalt);
            //Sleep(100);

             // Display FPS
            char fpsText[32];
            snprintf(fpsText, sizeof(fpsText), "FPS: %.2f", 1 / deltaTime);
            SetTextColor(hdc, RGB(255, 0, 0)); // Red text
            // SetBkColor(hdc, RGB(0,0,0));
            SetBkMode(hdc, TRANSPARENT); // Transparent background
            TextOut(hdc, 10, 10, fpsText, strlen(fpsText));
            EndPaint(hwnd, &ps);

            InvalidateRect(hwnd, NULL, TRUE);
            return 0;
        }
        case WM_TIMER: {
            // Invalidate the window's client area to trigger a repaint
            InvalidateRect(hwnd, NULL, TRUE);
        } break;
        case WM_DESTROY:
            PostQuitMessage(0);
            FreeCudaResources();
            return 0;
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }

    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    srand(time(NULL));
    const char CLASS_NAME[] = "PixelWindowClass";
    QueryPerformanceCounter(&startTime);
    WNDCLASS wc = {0};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    for (int m = 0; m < SIMHEIGHT*SIMWIDTH; m++)
        FluidArray[m] = CreateFluid(0.0, ((float)rand() / RAND_MAX)*2.0 - 1.0, ((float)rand() / RAND_MAX)*2.0 - 1.0);
    //FluidArray[37].mass = 1.0;
    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(
        0, CLASS_NAME, "Pixel Window",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
        width, height, NULL, NULL, hInstance, NULL
    );

    if (!hwnd) {
        return 0;
    }
    SetTimer(hwnd, 1, 0, NULL); // timer to invalidate window
    
    ShowWindow(hwnd, nCmdShow);

    MSG msg = {0};
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}
