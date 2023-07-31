#include "constants.h"
#include "stats.h"
#include <float.h>


double stddev(const double a[], const int size) {
    double m = mean(a, size);
    double sd = 0.0;
    for (int i = 0; i < size; i++) {
        sd += pow(a[i] - m, 2);
    }
    sd = sqrt(sd / (size - 1));
    return sd;
}

int histcounts(double y[], const int size, int nBins, int binCounts[DATA_SIZE], double binEdges[DATA_SIZE+1]) 
{

     int i = 0;
    // check min and max of input array
    double minVal = DBL_MAX, maxVal=-DBL_MAX;
    for(int i = 0; i < size; i++)
    {
        // printf("histcountInput %i: %1.3f\n", i, y[i]);
        
        if (y[i] < minVal)
        {
            minVal = y[i];
        }
        if (y[i] > maxVal)
        {
            maxVal = y[i];
        }
    }
    
    // if no number of bins given, choose spaces automatically
    if (nBins <= 0){
        nBins = ceil((maxVal-minVal)/(3.5*stddev(y, size)/pow(size, 1/3.)));
    }

    // and derive bin width from it
    double binStep = (maxVal - minVal)/nBins;
    
    
    for(i = 0; i < nBins; i++) {
        binCounts[i] = 0;
    }

    for(i = 0; i < size; i++) {
        int binInd = (y[i]-minVal)/binStep;
        if(binInd < 0)
            binInd = 0;
        if(binInd >= nBins)
            binInd = nBins-1;
        binCounts[binInd] += 1;
    }

    for(i = 0; i < nBins+1; i++) {
        binEdges[i] = i * binStep + minVal;
    }

    /*
    // debug
    for(i=0;i<nBins;i++)
    {
        printf("%i: count %i, edge %1.3f\n", i, binCounts[i], binEdges[i]);
    }
    */

    return nBins;
}

data_t DN_HistogramMode_5(data_t y[DATA_SIZE])
{
    const int size = DATA_SIZE;
    // NaN check
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return NAN;
        }
    }
    
    //const int nBins = 5;
    
    int histCounts[DATA_SIZE];
    double binEdges[DATA_SIZE];
    
    histcounts(y, size, 5, &histCounts, &binEdges);
    
    /*
    for(int i = 0; i < nBins; i++){
        printf("histCounts[%i] = %i\n", i, histCounts[i]);
    }
    for(int i = 0; i < nBins+1; i++){
        printf("binEdges[%i] = %1.3f\n", i, binEdges[i]);
    }
     */
    
    double maxCount = 0;
    int numMaxs = 1;
    double out = 0;;
    for(int i = 0; i < 5; i++)
    {
        // printf("binInd=%i, binCount=%i, binEdge=%1.3f \n", i, histCounts[i], binEdges[i]);
        
        if (histCounts[i] > maxCount)
        {
            maxCount = histCounts[i];
            numMaxs = 1;
            out = (binEdges[i] + binEdges[i+1])*0.5;
        }
        else if (histCounts[i] == maxCount){
            
            numMaxs += 1;
            out += (binEdges[i] + binEdges[i+1])*0.5;
        }
    }
    out = out/numMaxs;
    
    // arrays created dynamically in function histcounts
    free(histCounts);
    free(binEdges);
    
    return out;
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
    result = DN_HistogramMode_5(window);

    /* Pushing to FIFO Stream */
    output << result;
}
