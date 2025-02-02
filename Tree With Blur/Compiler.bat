nvcc -c Blur.cu -o Blur.o
gcc -c Drawer.c -o Drawer.o -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include"-lgdi32 -luser32
nvcc -o Tree.exe Drawer.o Blur.o -L"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\12.6\lib\x64" -lgdi32 -luser32
start Tree.exe



timeout /t 30