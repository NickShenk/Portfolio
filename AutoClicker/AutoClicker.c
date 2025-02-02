#include <windows.h>
#include <stdio.h>


void SimulateClick() {
    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
    Sleep(10); // 10 ms delay
    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
}

int main() {
    printf("Press F6 to start/stop the autoclicker. Press F7 to exit.\n");
    int running = FALSE;
    int delay = 0;
    while (TRUE) {
        if (GetAsyncKeyState(VK_F6) & 0x8000) {
            running = !running;
            printf("Autoclicker %s.\n", running ? "started" : "stopped");
            Sleep(500); // 500 ms delay between switching
            delay = 0;
        }
        if (GetAsyncKeyState(VK_F7) & 0x8000) { // checks if the most significant bit exist, if it does then the AND becomes non zero and if statement is true
            printf("Exiting program.\n");
            break;
        }

        if (running) {
            if (delay > 50) {// 500 ms delay between clicks
                SimulateClick();
                delay = -1;
            }
            delay++;
             
        }
        Sleep(10); // Reduce CPU usage
        
    }

    return 0;
}
