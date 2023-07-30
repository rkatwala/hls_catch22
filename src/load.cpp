#include "constants.h"

extern "C" void load(data_t* input, hls::stream<data_t> &output) {
    #pragma HLS INTERFACE mode=m_axi port=input bundle=gmem0
    #pragma HLS INTERFACE mode=axis port=&output
    
    static int counter = 0;
    data_t temp; 
    temp = input[counter];
    counter+=1; 
    output << temp;

}
