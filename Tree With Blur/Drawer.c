#include <windows.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For time()

// CUDA Kernel Declaration
extern void LaunchCudaKernel(COLORREF* oldscreen, COLORREF* screen, int width, int height, int minx, int miny, int maxx, int maxy);
extern  void FreeCudaResources();

static HBITMAP hBitmap = NULL;
static HBITMAP screenBitmap = NULL;
static void* pixelData = NULL;
static int height = 0;
static int width = 0;
static int minx;
static int miny;
static int maxx;
static int maxy;
COLORREF* CloneScreen(HDC hScreenDC) {
    // Create a compatible bitmap
    HBITMAP hBitmap = CreateCompatibleBitmap(hScreenDC, width, height);
    if (!hBitmap) {
        printf("Failed to create compatible bitmap.\n");
        return NULL;
    }

    // Create a compatible device context
    HDC hMemoryDC = CreateCompatibleDC(hScreenDC);
    if (!hMemoryDC) {
        printf("Failed to create compatible device context.\n");
        DeleteObject(hBitmap);
        return NULL;
    }

    // Select the bitmap into the memory DC
    HBITMAP hOldBitmap = (HBITMAP)SelectObject(hMemoryDC, hBitmap);

    // Bit-block transfer the screen content into the memory DC
    if (!BitBlt(hMemoryDC, 0, 0, width, height, hScreenDC, 0, 0, SRCCOPY)) {
        printf("BitBlt failed.\n");
        SelectObject(hMemoryDC, hOldBitmap);
        DeleteObject(hBitmap);
        DeleteDC(hMemoryDC);
        return NULL;
    }

    // Allocate memory for pixel data
    BITMAPINFO bmi = {0};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biHeight = -height;  // Top-down image
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;    // 32 bits per pixel
    bmi.bmiHeader.biCompression = BI_RGB;

    COLORREF* pixels = (COLORREF*)malloc(width * height * sizeof(COLORREF));
    if (!pixels) {
        printf("Failed to allocate memory for pixel data.\n");
        SelectObject(hMemoryDC, hOldBitmap);
        DeleteObject(hBitmap);
        DeleteDC(hMemoryDC);
        return NULL;
    }

    // Retrieve the pixel data
    if (!GetDIBits(hMemoryDC, hBitmap, 0, height, pixels, &bmi, DIB_RGB_COLORS)) {
        printf("GetDIBits failed.\n");
        free(pixels);
        pixels = NULL;
    }

    // Clean up
    SelectObject(hMemoryDC, hOldBitmap);
    DeleteObject(hBitmap);
    DeleteDC(hMemoryDC);

    return pixels;
}


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

void DrawWithCuda(HDC hdc,HDC screen, RECT clientRect) {
    if (!hBitmap || !pixelData || (clientRect.right - clientRect.left != width) || (clientRect.bottom - clientRect.top != height)) {
        InitializeDIBSection(hdc, clientRect);
    }
    COLORREF* clonedScreen = CloneScreen(hdc);
    
    // Call the CUDA kernel to fill the pixel data
    LaunchCudaKernel(clonedScreen, (COLORREF*)pixelData, width, height, minx, miny, maxx, maxy);
    free(clonedScreen);
    // Create a memory DC and select the bitmap
    HDC hdcMem = CreateCompatibleDC(hdc);
    HBITMAP hOldBitmap = (HBITMAP)SelectObject(hdcMem, hBitmap);
    // Copy the memory DC to the screen
    // BitBlt(hdc, 0, 0, width, height, hdcMem, 0, 0, SRCCOPY);
    // copy to the screen
    BitBlt(screen, 0, 0, width, height, hdcMem, 0, 0, SRCCOPY);
    // Cleanup
    SelectObject(hdcMem, hOldBitmap);
    DeleteObject(hOldBitmap);
    DeleteDC(hdcMem);
}






typedef struct colorGradient
{
    int rg;
    int gg;
    int bg;
} colorGradient;

colorGradient constructGradient(int r, int g, int b) {
    colorGradient temp;
    temp.rg = r;
    temp.gg = g;
    temp.bg = b;
    return temp;
}

double radians(double a) {
    return (a * M_PI) / 180.0;
}
double degrees(double a) {
    return (a *  180.0) /M_PI;
}
double angle(double a, double b) {
    if (a < 0) return 180 + degrees(atan(b/a));
    if (a == 0) {
      if (b > 0) return 90;
      return 270 ;
    }
   return degrees(atan(b/a));
}
int generateRandom(int min, int max) {
    return (rand() % (max - min)) + min;
}
void line(HDC hdc, int x1, int y1, int x2, int y2, COLORREF color, int thickness) {
    // Set the pen for the line
    HPEN hPen = CreatePen(PS_SOLID, thickness, color);
    HPEN hOldPen = SelectObject(hdc, hPen);

    // Set the starting point of the line
    MoveToEx(hdc, x1, y1, NULL);

    // Draw the line to the specified endpoint
    LineTo(hdc, x2, y2);

    // Restore the original pen
    SelectObject(hdc, hOldPen);
    DeleteObject(hPen);
}
COLORREF changeColor(COLORREF color, colorGradient colorgradient) {
    // Extract the current RGB components from the COLORREF
    int r = GetRValue(color); // Extract the red component
    int g = GetGValue(color); // Extract the green component
    int b = GetBValue(color); // Extract the blue component
    // Modify the components with the gradients(rg, gg, bg) and clamp to [0, 255]
    r = max(0, min(255, r + colorgradient.rg));
    g = max(0, min(255, g + colorgradient.gg));
    b = max(0, min(255, b + colorgradient.bg));
    return RGB(r, g, b);
}

RECT rect;
typedef struct tree {
    int level;
    int x;
    int y;
    double m;
    double v;
    double va;
    double a;
    double aa;
    int scale;
    COLORREF color;
    colorGradient lcolor;
    colorGradient rcolor;
} tree;
typedef struct Node {
    tree call;
    struct Node* next;
} Node;
Node* head = NULL;
Node* tail = NULL;
void push(tree item) {
    Node* temp = (Node*)malloc(sizeof(Node));
    if (temp == NULL) {
        exit(1);
    }
    temp->call = item;
    temp->next = NULL;
    if (head == NULL) {
        head = temp;
        tail = temp;
    } else {
        tail->next = temp;
        tail = temp;
    }
}
tree pop() {
    if (head != NULL) {
        tree out = head->call;
        Node* temp = head;
        head = head->next;
        if (head == NULL) {
            tail = NULL;
        }
        free(temp);
        return out;
    }
    
}
void clearQ() {
    while (head != NULL) pop();
    return;
}
void recTree(HDC hdc,HDC screen, tree call) {
    if ((call.level == 1) || (call.m <= 0)) return; // base case
    int scalesqrd = pow(call.scale,2);
    int endx = ((cos(radians(call.aa))*call.a)/2)*(scalesqrd) + (call.v*cos(radians(call.va))*call.scale) + call.x;
    int endy = ((sin(radians(call.aa))*call.a)/2)*(scalesqrd) + (call.v*sin(radians(call.va))*call.scale) + call.y;
    if (call.a != 0) {
        double vx = cos(radians(call.aa))*call.a/2 + call.v*cos(radians(call.va));
        double vy = sin(radians(call.aa))*call.a/2 + call.v*sin(radians(call.va));
        call.v = pow((pow(vx,2)+pow(vy,2)),.5);
        call.va = angle(vx,vy);
    }
    if (minx > endx) minx = endx;
    else if (maxx < endx) maxx = endx;
    if (miny > endy) miny = endy;
    else if (maxy < endy) maxy = endy;
    line(hdc,call.x,call.y,endx,endy,call.color, call.m);
    // Step 4: Run DrawWithCuda to process the offscreen buffer
    double massdecline = .75;
    double velocitydecline = .9;
    int rexists = 0;
    int lexists = 0;
    tree rchild;
    tree lchild;
    // right Child
    if (generateRandom(0, call.level) != 0) {
        rexists = 1;
        rchild.level = call.level - 1;
        rchild.x = endx;
        rchild.y = endy;
        rchild.m = call.m*massdecline;
        rchild.v = call.v*velocitydecline;
        rchild.va = call.va + generateRandom(15,30);
        rchild.a = call.a;
        rchild.aa = call.aa;
        rchild.scale = call.scale;
        rchild.color = changeColor(call.color,call.rcolor);
        rchild.rcolor = call.rcolor;
        rchild.lcolor = call.lcolor;
    } if (generateRandom(0, call.level) != 0) {
        // left Child
        lexists = 1;
        lchild.level = call.level - 1;
        lchild.x = endx;
        lchild.y = endy;
        lchild.m = call.m*massdecline;
        lchild.v = call.v*velocitydecline;
        lchild.va = call.va - generateRandom(15,30);
        lchild.a = call.a;
        lchild.aa = call.aa;
        lchild.scale = call.scale;
        lchild.color = changeColor(call.color,call.lcolor);
        lchild.rcolor = call.rcolor;
        lchild.lcolor = call.lcolor;
    }
    if (generateRandom(0,2)) {
        if (lexists) push(lchild);
        if (rexists) push(rchild);
    } else {
        if (rexists) push(rchild);
        if (lexists) push(lchild);
    }
}


// Window procedure function prototype
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    srand(time(NULL));
    const char CLASS_NAME[] = "PixelWindowClass";
    // Define a window class
    WNDCLASS wc = {0};
    wc.lpfnWndProc = WindowProc;         // Window procedure function
    wc.hInstance = hInstance;           // Handle to the application instance
    wc.lpszClassName = CLASS_NAME;      // Name of the window class

    // Register the window class
    RegisterClass(&wc);
    // Create the window
    HWND hwnd = CreateWindowEx(
        0,                              // Optional window styles
        CLASS_NAME,                     // Window class name
        "Pixel Window",                 // Window title
        WS_OVERLAPPEDWINDOW,            // Window style
        CW_USEDEFAULT, CW_USEDEFAULT,   // Position
        800, 600,                       // Size
        NULL,                           // Parent window
        NULL,                           // Menu
        hInstance,                      // Instance handle
        NULL                            // Additional application data
    );

    if (!hwnd) {
        return 0; // If the window creation failed
    }

    // Show the window
    ShowWindow(hwnd, nCmdShow);

    // Run the message loop
    MSG msg = {0};
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg); // pass to WindowProc
    }

    return 0;
}

// Window procedure function
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_KEYDOWN:
            if (wParam == 'F') {
                InvalidateRect(hwnd, NULL, TRUE); 
            }
            break;
        case WM_SIZE:
            InvalidateRect(hwnd, NULL, TRUE);  // Invalidate the window to force a redraw
            break;
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            
            GetClientRect(hwnd, &rect);
            int width = rect.right - rect.left;
            int height = rect.bottom - rect.top;
            // Step 1: Create an offscreen memory DC and bitmap
            HDC hdcMem = CreateCompatibleDC(hdc);
            HBITMAP hBitmap = CreateCompatibleBitmap(hdc, width, height);
            HBITMAP hOldBitmap = (HBITMAP)SelectObject(hdcMem, hBitmap);

            // Step 2: Fill the background in the offscreen buffer
            HBRUSH hBrush = CreateSolidBrush(RGB(0, 0, 0)); // Background Color
            FillRect(hdcMem, &rect, hBrush);
            DeleteObject(hBrush);
            

            int gradientExtreme = 60;
            clearQ();
            // root 1
            tree root;
            root.level = 19;
            root.x = width/2;
            minx = root.x;
            maxx = root.x;
            root.y = height*(.7);
            miny = root.y;
            maxy = root.y;
            root.m = 6;
            root.v = 1;
            root.va = 270;
            root.a = 0.005;
            root.aa = generateRandom(0,361);
            root.scale = 75;
            root.color = RGB(rand(), rand(), rand());
            root.rcolor = constructGradient((rand()%gradientExtreme) - gradientExtreme/2,
                                            (rand()%gradientExtreme) - gradientExtreme/2,
                                            (rand()%gradientExtreme) - gradientExtreme/2);
            root.lcolor = constructGradient((rand()%gradientExtreme) - gradientExtreme/2,
                                            (rand()%gradientExtreme) - gradientExtreme/2,
                                            (rand()%gradientExtreme) - gradientExtreme/2);
            push(root);
            int currentlevel = 19;
            while (head != NULL) {
                if (head->call.level == currentlevel) {
                    DrawWithCuda(hdcMem,hdc, rect);
                    currentlevel -= 1;
                }
                recTree(hdcMem,hdc,pop());
            }
            // DrawWithCuda(hdcMem,hdc, rect);
            // Cleanup
            SelectObject(hdcMem, hOldBitmap);
            DeleteObject(hBitmap);
            DeleteDC(hdcMem);
            EndPaint(hwnd, &ps);
            ReleaseDC(NULL, hdc);
            FreeCudaResources();
            // Sleep(3000); // sleep for 3 seconds
            // InvalidateRect(hwnd, NULL, TRUE);
            return 0;
        }
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}