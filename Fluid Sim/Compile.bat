nvcc -c Cuda.cu -o Cuda.o
gcc -c Sim.c -o Sim.o -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include"-lgdi32 -luser32
nvcc -o Sim.exe Sim.o Cuda.o -L"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\12.6\lib\x64" -lgdi32 -luser32



timeout /t 30
timeout /t 25