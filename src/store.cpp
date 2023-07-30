#include "constants.h"

extern "C" void store(hls::stream<data_t> &input, data_t* output) {
    #pragma HLS INTERFACE mode=axis port=&input
    #pragma HLS INTERFACE mode=m_axi port=output bundle=gmem1
    
    static int counter = 0;
    data_t temp; 
    input >> temp;
    output[counter] = temp;
    counter+=1; 
}
