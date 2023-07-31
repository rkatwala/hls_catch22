#include "host.h"
#include "constants.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include "stats.h"
#include <float.h>

/* Orginal code from C22 https://github.com/DynamicsAndNeuralSystems/catch22/tree/main */
// double DN_Mean(const double a[], const int size)
// {
//     double m = 0.0;
//     for (int i = 0; i < size; i++) {
//         m += a[i];
//     }
//     m /= size;
//     return m;
// }

/* Modified */
double mean(data_t a[DATA_SIZE], const int size)
{
    double m = 0.0;
    for (int i = 0; i < size; i++) {
        m += a[i];
    }
    m /= size;
    return m;
}

double stddev(data_t a[DATA_SIZE], const int size) {
    double m = mean(a, size);
    double sd = 0.0;
    for (int i = 0; i < size; i++) {
        sd += pow(a[i] - m, 2);
    }
    sd = sqrt(sd / (size - 1));
    return sd;
}

int histcounts(data_t y[DATA_SIZE], const int size, int nBins, int* binCounts, double* binEdges) 
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
    
    int histCounts[5];
    double binEdges[6];
    
    histcounts(y, size, 5, histCounts, binEdges);
    
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
    // free(histCounts);
    // free(binEdges);
    
    return out;
}

int main(int argc, char** argv) {


    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }
   
    clock_t htod, dtoh, comp;


    /*====================================================CL===============================================================*/
    std::string binaryFile = argv[1];
    cl_int err;
    cl::Context context;
    cl::Kernel krnl1, krnl2, krnl3;
    cl::CommandQueue q;
   
    auto devices = get_xil_devices();
    auto fileBuf = read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, 0, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            std::cout << "Setting CU(s) up..." << std::endl;
            OCL_CHECK(err, krnl1 = cl::Kernel(program, "krnl", &err));
            OCL_CHECK(err, krnl2 = cl::Kernel(program, "load", &err));
            OCL_CHECK(err, krnl3 = cl::Kernel(program, "store", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }


    /*====================================================INIT INPUT/OUTPUT VECTORS===============================================================*/
    std::vector<data_t, aligned_allocator<data_t> > input(TOTAL_DATA_SIZE);
    data_t *input_sw = (data_t*) malloc(sizeof(data_t) * (TOTAL_DATA_SIZE + DATA_SIZE));
    std::vector<data_t, aligned_allocator<data_t> > output_hw(TOTAL_DATA_SIZE - DATA_SIZE);
    data_t *output_sw = (data_t*) malloc(sizeof(data_t) * (TOTAL_DATA_SIZE - DATA_SIZE));

    for (int i = 0; i < DATA_SIZE - 1; i++) {
        input_sw[i] = 0.0;
    }

    int lb = 1, ub = 99999999;
    srand(time(nullptr));
    for (int i = 0; i < TOTAL_DATA_SIZE; i++) {
        int temp = (rand() % (ub - lb + 1)) + lb;
        input[i] = temp;
        input_sw[i + DATA_SIZE - 1] = temp;
    }

    /*====================================================SW VERIFICATION===============================================================*/
    for (int j = 0; j < TOTAL_DATA_SIZE - DATA_SIZE; j++) {
        output_sw[j] = DN_HistogramMode_5(input_sw + j);
    }
    /*====================================================Setting up kernel I/O===============================================================*/

    /* INPUT BUFFERS */
    OCL_CHECK(err, cl::Buffer buffer_input(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * TOTAL_DATA_SIZE, input.data(), &err));  


    /* OUTPUT BUFFERS */
    OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * (TOTAL_DATA_SIZE - DATA_SIZE), output_hw.data(), &err));


    /* SETTING INPUT PARAMETERS */
    OCL_CHECK(err, err = krnl2.setArg(0, buffer_input));
    OCL_CHECK(err, err = krnl3.setArg(1, buffer_output));

    /*====================================================KERNEL===============================================================*/
    /* HOST -> DEVICE DATA TRANSFER*/
    std::cout << "HOST -> DEVICE" << std::endl;
    htod = clock();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input}, 0 /* 0 means from host*/));
    q.finish();
    htod = clock() - htod;
   
    /*STARTING KERNEL(S)*/
    std::cout << "STARTING KERNEL(S)" << std::endl;
    comp = clock();
    for (int i = 0; i < TOTAL_DATA_SIZE - DATA_SIZE; i++) {
        OCL_CHECK(err, err = q.enqueueTask(krnl2));
        q.finish();
        OCL_CHECK(err, err = q.enqueueTask(krnl1));
        q.finish();
        OCL_CHECK(err, err = q.enqueueTask(krnl3));
        q.finish();
    }
    comp = clock() - comp;
    std::cout << "KERNEL(S) FINISHED" << std::endl;


    /*DEVICE -> HOST DATA TRANSFER*/
    std::cout << "HOST <- DEVICE" << std::endl;
    dtoh = clock();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    dtoh = clock() - dtoh;


    /*====================================================VERIFICATION & TIMING===============================================================*/

    printf("Host -> Device : %lf ms\n", 1000.0 * htod/CLOCKS_PER_SEC);
    printf("Device -> Host : %lf ms\n", 1000.0 * dtoh/CLOCKS_PER_SEC);
    printf("Computation : %lf ms\n",  1000.0 * comp/CLOCKS_PER_SEC);
   
    bool match = true;
    for (int i = 0; i < TOTAL_DATA_SIZE - DATA_SIZE; i++) {
        std::cout << "i = " << i << " HW: " << output_hw[i] << " vs " << "SW: " << output_sw[i] << std::endl;
        if (output_hw[i] != output_sw[i]) {
            
            match = false;
        }
    }
    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;


    free(output_sw);
    free(input_sw);


    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}

