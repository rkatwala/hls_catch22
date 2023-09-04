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
// // double mean(const double a[], const int size)
// {
//     double m = 0.0;
//     for (int i = 1; i < size; i++) {
//         a[0] += a[i];
//     }
//     return a[0]/size;
//}

data_t findMinimumReduction_Mean(data_t window[DATA_SIZE]) {
    data_t y[DATA_SIZE];
    min_reduct_copy_loop : for (int i = 0; i < DATA_SIZE; i++) {
        #pragma HLS UNROLL
        y[i] = window[i];
    }
    min_reduct_step_loop : for (int s = 1; s <= DATA_SIZE/2; s*=2) {
        #pragma HLS LOOP_TRIPCOUNT min=8 max=8
        #pragma HLS PIPELINE
        compute_mean_loop : for (int i = 0; i < DATA_SIZE; i+=2*s) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=64
            #pragma HLS UNROLL
            //y[i] = (y[i] < y[i + s]) ? y[i]: y[i + s];
            y[i] = y[i + s];
        }
    }
   // return y[0];
   return y[0]/DATA_SIZE;
}
data_t stddev(data_t a[DATA_SIZE])
{
    data_t m = findMinimumReduction_Mean(a);
    data_t sd = 0.0;
    for (int i = 0; i < DATA_SIZE; i++) {
        sd += pow(a[i] - m, 2);
    }
    sd = sqrt(sd / (DATA_SIZE - 1));
    return sd;
}

// data_t FC_LocalSimple_mean3_stderr(data_t y[DATA_SIZE])
// {
    
//         // NaN check
//     int train_length = 3;
//     int size = DATA_SIZE;   
//     FC_LocalSimple_mean3_stderr_loop1: for(int i = 0; i < size; i++)
//     {
//         if(isnan(y[i]))
//         {
//             return NAN;
//         }
//     }
    
//     //double* res = new double[size - train_length];
//      data_t res[DATA_SIZE- 3];
    
//     for (int i = 0; i < size - 3; i++)
//     {
//         double yest = 0;
//         for (int j = 0; j < 3; j++)
//         {
//             yest += y[i+j];
            
//         }
//         yest /= 3;
        
//         res[i] = y[i+3] - yest;
//     }
    
//     data_t output = stddev(res);
    
//     //free(res);
//     return output;
// }


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
    result = findMinimumReduction_Mean(window);

    /* Pushing to FIFO Stream */
    output << result;
}

