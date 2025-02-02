#include <stdio.h>
#include <stdlib.h>
#include <math.h>
float sig(float a) {
    if (a > 10.0) a = 10.0;
    if (a < -10.0) a = -10.0;
    return 2.0/(1.0+ pow(1.5,-a)) - 1.0;
}   
int max(int* shape, int layers) {
    int max = 0;
    for (int i = 0; i < layers; i++) {
        if (shape[i] > max) {
            max = shape[i];
        }
    }
    return max;
}
void think(float* inp, float* weights, int* shape, int layers, float* buffer) {
    int maxlayer = max(shape,layers);
    float *arr1 = (float*)malloc(maxlayer * sizeof(float));
    for (int i = 0; i < shape[0]; i++) arr1[i] = inp[i];
    float *arr2 = (float*)malloc(maxlayer * sizeof(float));
    int offset = 0;
    int lastm = shape[0];
    int currm = 0;
    float sum;
    float *temp;
    for (int m = 1; m < layers; m++) {
        currm = shape[m];
        for (int x = 0; x < currm; x++) {
            sum = weights[offset];
            for (int y = 0; y < lastm; y++) {
                sum += weights[offset + y + 1] * arr1[y];
            }
            offset += 1 + lastm;
            
            if (sum >= 0 || x == (currm - 1)) { // leakyRelu
                if (sum > 100.0) sum = 100.0;
                arr2[x] = sum;
            } else {
                if (sum < -100.0) sum = -100.0;
                arr2[x] = sum*0.1;
            }
        }
        temp = arr1;
        arr1 = arr2;
        arr2 = temp;
        lastm = currm;
    }
    free(arr2);
    for (int i = 0; i < lastm; i++) {
        buffer[i] = sig(arr1[i]); // return as a buffer to free all memory
    }
    free(arr1);
}
void fitness(float* bot, float* inputs, int days, float* prices, int* shape, int layers, float* output) {
    float feedback = 0.0;
    float decision = 0.0;
    const int inputcount = shape[0];
    float *input = (float*)malloc(inputcount * sizeof(float));
    float bought = 0.0;
    float* choice = (float*)malloc(shape[layers - 1] * sizeof(float));
    float sum = 0.0;
    *output = 0.0;
    for (int day = 1; day < days; day++) {
        // printf("%f %n", prices[day]);
        for (int i = 0; i < 6; i++) {
            input[i] = inputs[i+(day - 1)*6];
            sum += inputs[i+(day - 1)*6];
        }
        input[inputcount - 1] = feedback;
        think(input, bot, shape, layers, choice);
        decision = choice[0];
        /*
        if (decision > 0 && feedback <= 0) { // BUY
            bought = prices[day] * 1.005;
        } else if (decision <= 0 && feedback > 0) { // SELL
            *output += ((prices[day] * .995) - bought)/bought;
        }
        */
       if ((prices[day] - prices[day - 1]) > 0) {
            *output += decision;
       } else {
            *output -= decision;
       }
       if (feedback * decision < 0) {
        *output *= .995; // punishment for buying/selling
       }
        feedback = decision;

    }
    free(choice);
    free(input);
    // if (feedback > 0) *output += ((prices[days - 1]* .995) - bought) / bought; // force sell if bought in at the end
}
