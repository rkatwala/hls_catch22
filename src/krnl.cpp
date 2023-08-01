#include "constants.h"
#include "stats.h"
/*

    From Catch 22 GitHub: https://github.com/DynamicsAndNeuralSystems/catch22/tree/main

    in C/DN_Mean.cpp

    Modified to work with HLS


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
*/
double mean(const double a[], const int size)
{
    double m = 0.0;
    for (int i = 0; i < size; i++) {
        m += a[i];
    }
    m /= size;
    return m;
}
double stddev(const double a[], const int size)
{
    double m = mean(a, size);
    double sd = 0.0;
    for (int i = 0; i < size; i++) {
        sd += pow(a[i] - m, 2);
    }
    sd = sqrt(sd / (size - 1));
    return sd;
}

data_t FC_LocalSimple_mean3_stderr(data_t y[DATA_SIZE])
{
    
        // NaN check
    int train_length = 3;
    int size = DATA_SIZE;   
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return NAN;
        }
    }
    
    double* res = new double[size - train_length];
    
    for (int i = 0; i < size - train_length; i++)
    {
        double yest = 0;
        for (int j = 0; j < train_length; j++)
        {
            yest += y[i+j];
            
        }
        yest /= train_length;
        
        res[i] = y[i+train_length] - yest;
    }
    
    double output = stddev(res, size - train_length);
    
    //free(res);
    return output;
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
    result = FC_LocalSimple_mean3_stderr(window);

    /* Pushing to FIFO Stream */
    output << result;
}


