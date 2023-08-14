#include "host.h"
#include "constants.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include "stats.h"

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
/*double DN_Mean(data_t* a)
{
    double m = 0.0;
    for (int i = 0; i < DATA_SIZE; i++) {
        m += a[i];
    }
    m /= DATA_SIZE;
    return m;
}
*/
double mean(data_t a[DATA_SIZE], const int size)
{
    double m = 0.0;
    for (int i = 0; i < size; i++) {
        m += a[i];
    }
    m /= size;
    return m;
}
double stddev(data_t a[DATA_SIZE], const int size)
{
    double m = mean(a, size);
    double sd = 0.0;
    for (int i = 0; i < size; i++) {
        sd += pow(a[i] - m, 2);
    }
    sd = sqrt(sd / (size - 1));
    return sd;
}
double FC_LocalSimple_mean_stderr(data_t y[DATA_SIZE])
{
    // NaN check
    int size = DATA_SIZE;
    int train_length = 3;
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return NAN;
        }
    }
    
    double res[DATA_SIZE- train_length];

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

double FC_LocalSimple_mean3_stderr(data_t y[DATA_SIZE])
{
    //int size = DATA_SIZE;
    return FC_LocalSimple_mean_stderr(y);
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
        output_sw[j] = FC_LocalSimple_mean3_stderr(input_sw + j);
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

