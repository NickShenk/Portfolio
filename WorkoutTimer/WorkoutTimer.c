#include <Windows.h>
#include <mmsystem.h>  // For mciSendString
#include <stdio.h>

#pragma comment(lib, "winmm.lib")



int main() {
    // Open once
    mciSendString(
        "open \"Alarm.mp3\" type mpegvideo alias mySound",
        NULL, 0, NULL
    );

    int waittime;
    printf("How many seconds in between sets? ");
    scanf("%d", &waittime);

    while(TRUE) {
        printf("RAN\n");
        Sleep(waittime * 1000);

        // Just replay from start
        mciSendString("seek mySound to start", NULL, 0, NULL);
        mciSendString("play mySound", NULL, 0, NULL);
    }

    // If you ever break out of the loop:
    mciSendString("close mySound", NULL, 0, NULL);
    return 0;
}
