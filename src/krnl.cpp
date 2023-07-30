#include "constants.h"
/*

    From Catch 22 GitHub: https://github.com/DynamicsAndNeuralSystems/catch22/tree/main

    in C/DN_Mean.cpp

    Modified to work with HLS

*/
data_t DN_Mean(data_t a[DATA_SIZE])
{
    #pragma HLS INLINE
    double m = 0.0;
    for (int i = 0; i < DATA_SIZE; i++) {
        #pragma HLS PIPELINE
        m += a[i];
    }
    m /= DATA_SIZE;
    return m;
}

extern "C" void krnl(hls::stream<data_t> &input, hls::stream<data_t> &output) {
    #pragma HLS INTERFACE mode=axis port=&input
    #pragma HLS INTERFACE mode=axis port=&output
    
    static data_t window[DATA_SIZE];
    data_t result;

    /* Shifting window to the left, newest data point is last index */
    for (int i = 0; i < DATA_SIZE-1; i++) {
        #pragma HLS UNROLL
        window[i] = window[i+1];
    }
    input >> window[DATA_SIZE-1];
    
    /* Feature Extraction */
    result = DN_Mean(window);

    /* Pushing to FIFO Stream */
    output << result;
}


