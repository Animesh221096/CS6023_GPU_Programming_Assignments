// Your solution. This is the only file you submit.
// It is compiled together with the provided harness.cu:
//     nvcc -arch=sm_70 harness.cu wildfire.cu -o main
// See wildfire.h for the definitions you can use and the full contract of
// run_simulation().
#include "wildfire.h"

#define BLOCK_SIZE 256 // you can change it.

// ============================================================================
// CUDA KERNELS
// ============================================================================












// ============================================================================
// SIMULATION DRIVER
// ============================================================================

void run_simulation(Params p, uint8_t *state, uint8_t *fuel, const uint8_t *elev,
                    int F, const int *fires, int K, const int *stations,
                    int C, int *crews)
{
    /*
    ToDo








    */
}
