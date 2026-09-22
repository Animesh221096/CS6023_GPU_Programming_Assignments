#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>

#define INF 2147483647 // simply INT_MAX
#define BLOCK_SIZE 256 // you can change it.


// ============================================================================
// CUDA KERNELS
// ============================================================================

// Kernel to initialize distances: dist[source] = 0, all others = INF
__global__ void initialize_distances(int *dist, int N, int source)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        dist[idx] = (idx == source) ? 0 : INF;
    }
}

// Kernel to relax edges in parallel
// Each thread processes one vertex u and relaxes all its outgoing edges
__global__ void relax_edges(
    int *dist,
    int *offsets,
    int *neighs,
    int *weights,
    int N,
    int *updated)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N)
        return;

    // Skip if source vertex is unreachable
    if (dist[u] == INF)
        return;

    // Relax all outgoing edges from u: (u, v) where v ∈ neighs[offsets[u]...offsets[u+1])
    for (int edge_idx = offsets[u]; edge_idx < offsets[u + 1]; ++edge_idx)
    {
        int v = neighs[edge_idx];
        int weight = weights[edge_idx];
        int new_dist = dist[u] + weight;

        // Use atomic minimum to safely update dist[v]
        // This ensures that concurrent updates from different threads don't race
        if (new_dist < dist[v])
        {
            int old_val = atomicMin(&dist[v], new_dist);
            if (old_val > new_dist)
            {
                *updated = 1;
            }
        }
    }
}

// ============================================================================
// SINGLE SOURCE DELTA-STEPPING DRIVER
// ============================================================================

void run_delta_stepping_single_source(
    /*




    */
)
{
    /*
    for adaptive delta, print from within this function. That would be easy.


    */
}


// ============================================================================
// SINGLE SOURCE BELLMAN-FORD DRIVER
// ============================================================================

void run_parallel_bellman_ford_single_source(
    int *d_dist,
    int *d_offsets,
    int *d_neighs,
    int *d_weights,
    int N,
    int E,
    int source)
{
    // Calculate grid and block dimensions
    int threads_per_block = BLOCK_SIZE;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    // Phase 1: Initialize distances
    initialize_distances<<<blocks, threads_per_block>>>(d_dist, N, source);
    cudaDeviceSynchronize();

    // Allocate device memory for the "updated" flag
    int *d_updated = nullptr;
    cudaMalloc(&d_updated, sizeof(int));

    // Phase 2: Iteratively relax edges until convergence
    // Bellman-Ford worst case: N-1 iterations, but may converge earlier
    for (int iter = 0; iter < N - 1; ++iter)
    {
        // Reset the updated flag to 0 before this iteration
        int h_updated = 0;
        cudaMemcpy(d_updated, &h_updated, sizeof(int), cudaMemcpyHostToDevice);

        // Launch relaxation kernel
        relax_edges<<<blocks, threads_per_block>>>(
            d_dist,
            d_offsets,
            d_neighs,
            d_weights,
            N,
            d_updated);
        cudaDeviceSynchronize();

        // Check if any updates occurred in this iteration
        cudaMemcpy(&h_updated, d_updated, sizeof(int), cudaMemcpyDeviceToHost);

        // If no updates, the algorithm has converged early
        if (h_updated == 0)
        {
            break;
        }
    }

    cudaFree(d_updated);
}


// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    std::ifstream infile(argv[1]);
    if (!infile.is_open())
    {
        std::cerr << "Error: Unable to open input file " << argv[1] << "\n";
        return 1;
    }

    std::ofstream outfile(argv[2]);
    if (!outfile.is_open())
    {
        std::cerr << "Error: Unable to open output file " << argv[2] << "\n";
        return 1;
    }

    int delta_mode, K, N, E, S_count;
    infile >> delta_mode >> K;
    infile >> N >> E >> S_count;

    std::vector<int> sources(S_count);
    for (int i = 0; i < S_count; ++i)
    {
        infile >> sources[i];
    }

    int *offsets = new int[ N + 1 ] { 0 };
    for (int i = 0; i <= N; ++i)
    {
        infile >> offsets[i];
    }

    int *neighs = new int[ E ] { 0 };
    for (int i = 0; i < E; ++i)
    {
        infile >> neighs[i];
    }

    int *weights = new int[ E ] { 0 };
    for (int i = 0; i < E; ++i)
    {
        infile >> weights[i];
    }

    infile.close();

    int *d_offsets = nullptr, *d_neighs = nullptr, *d_weights = nullptr;
    cudaMalloc(&d_offsets, (N + 1) * sizeof(int));
    cudaMalloc(&d_neighs, E * sizeof(int));
    cudaMalloc(&d_weights, E * sizeof(int));

    cudaMemcpy(d_offsets, offsets, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighs, neighs, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, E * sizeof(int), cudaMemcpyHostToDevice);

    int *h_tent = new int[ N ] { 0 }; // sssp distance array. tent means tentative distance
    int *d_tent;
    cudaMalloc(&d_tent, N * sizeof(int));


    /*
    ToDo








    */

    // Process each source query sequentially
    for (int i = 0; i < S_count; ++i)
    {
        int source = sources[i];

        outfile << source << "\n";

        run_delta_stepping_single_source(
            /*
            ToDo
            





            */
        );

        // Run parallel Bellman-Ford for this source
        run_parallel_bellman_ford_single_source(
            d_tent,
            d_offsets,
            d_neighs,
            d_weights,
            N,
            E,
            source);

        // Copy results back to host


        cudaMemcpy(h_tent, d_tent, N * sizeof(int), cudaMemcpyDeviceToHost);

        // Write source vertex
        outfile << source << "\n";

        // Write distances for all vertices
        for (int v = 0; v < N; ++v)
        {
            outfile << h_tent[v] << "\n";
        }
    }

    // Cleanup
    cudaFree(d_offsets);
    cudaFree(d_neighs);
    cudaFree(d_weights);
    cudaFree(d_tent);

    delete[] offsets;
    delete[] neighs;
    delete[] weights;
    delete[] h_tent;

    outfile.close();
    return 0;
}