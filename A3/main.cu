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
        // Set source distance to 0, all others to infinity
        dist[idx] = (idx == source) ? 0 : INF;
    }
}

// Rebuild the Far queue: every vertex with a finite tentative distance
// strictly greater than last_cutoff is unsettled/active.
// (Equivalent to ghost pruning: entries with key <= last_cutoff are dropped.)
__global__ void build_far_queue(
    int *dist,
    int *far_vertices,
    int *count,
    int N,
    int last_cutoff)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < N && dist[u] != INF && dist[u] > last_cutoff)
    {
        // Add unsettled vertices (those with dist > cutoff) to queue
        int pos = atomicAdd(count, 1);
        far_vertices[pos] = u;
    }
}

// Fill the key array: key = current tentative distance of each far vertex
__global__ void fill_keys(int *dist, int *far_vertices, int *far_keys, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
    {
        // Store the distance of each vertex in the Far queue for sorting
        far_keys[i] = dist[far_vertices[i]];
    }
}

// Light edge relaxation (weight <= delta) from active vertices (dist <= cutoff)
__global__ void relax_light_edges(
    int *dist,
    int *offsets,
    int *neighs,
    int *weights,
    int N,
    int delta,
    int cutoff,
    int *updated)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N)
        return;

    int du = dist[u];
    // Only process vertices within current cutoff
    if (du == INF || du > cutoff)
        return;

    // Relax all light edges (weight <= delta) from this vertex
    for (int e = offsets[u]; e < offsets[u + 1]; ++e)
    {
        int w = weights[e];
        if (w > delta)
            continue; // Skip heavy edges

        int v = neighs[e];
        int nd = du + w;
        // Update neighbor distance if shorter path found
        if (nd < dist[v])
        {
            int old = atomicMin(&dist[v], nd);
            if (old > nd)
            {
                *updated = 1; // Flag that updates occurred
            }
        }
    }
}

// Heavy edge relaxation (weight > delta) from active vertices (dist <= cutoff)
__global__ void relax_heavy_edges(
    int *dist,
    int *offsets,
    int *neighs,
    int *weights,
    int N,
    int delta,
    int cutoff,
    int *updated)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N)
        return;

    int du = dist[u];
    // Only process vertices within current cutoff
    if (du == INF || du > cutoff)
        return;

    // Relax all heavy edges (weight > delta) from this vertex
    for (int e = offsets[u]; e < offsets[u + 1]; ++e)
    {
        int w = weights[e];
        if (w <= delta)
            continue; // Skip light edges

        int v = neighs[e];
        int nd = du + w;
        // Update neighbor distance if shorter path found
        if (nd < dist[v])
        {
            int old = atomicMin(&dist[v], nd);
            if (old > nd)
            {
                *updated = 1; // Flag that updates occurred
            }
        }
    }
}

// ============================================================================
// SINGLE SOURCE DELTA-STEPPING DRIVER
// ============================================================================

void run_delta_stepping_single_source(
    int *d_dist,
    int *d_offsets,
    int *d_neighs,
    int *d_weights,
    int N,
    int E,
    int source,
    int delta_mode,
    int K,
    int *d_far_vertices,
    int *d_far_keys,
    int *d_updated,
    int *d_count,
    std::ofstream &outfile)
{
    int threads_per_block = BLOCK_SIZE;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    // Initialize distances: source=0, others=INF
    initialize_distances<<<blocks, threads_per_block>>>(d_dist, N, source);
    cudaDeviceSynchronize();

    thrust::device_ptr<int> far_v_ptr(d_far_vertices);
    thrust::device_ptr<int> far_k_ptr(d_far_keys);

    int last_cutoff = -1;
    int round_num = 0;

    // MAIN LOOP - Continue until all reachable vertices are settled
    while (true)
    {
        round_num++;

        // =====================================================
        // PHASE 3: Rebuild Far queue (includes ghost pruning)
        // =====================================================

        // Reset counter before building new Far queue
        cudaMemset(d_count, 0, sizeof(int));

        // Build Far queue: collect all vertices with dist > last_cutoff
        build_far_queue<<<blocks, threads_per_block>>>(
            d_dist, d_far_vertices, d_count, N, last_cutoff);

        // Wait for all threads to finish atomic operations
        cudaDeviceSynchronize();

        // Retrieve count of unsettled vertices
        int far_count = 0;
        cudaMemcpy(&far_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

        // TERMINATION: If no unsettled vertices remain, we're done
        if (far_count == 0)
        {
            break;
        }

        // Fill keys array with distances of vertices in Far queue
        fill_keys<<<blocks, threads_per_block>>>(
            d_dist, d_far_vertices, d_far_keys, far_count);
        cudaDeviceSynchronize();

        // Sort Far queue by distance using thrust
        thrust::sort_by_key(far_k_ptr, far_k_ptr + far_count, far_v_ptr);
        cudaDeviceSynchronize();

        // Get minimum distance (Dmin) from sorted queue
        int Dmin = 0;
        cudaMemcpy(&Dmin, d_far_keys, sizeof(int), cudaMemcpyDeviceToHost);

        // Calculate Delta and Dcutoff based on mode
        int delta = 0;
        int Dcutoff = 0;

        if (delta_mode == -1)
        {
            // Adaptive mode: set delta based on K-th smallest distance
            int itarget = std::min(K - 1, far_count - 1);
            cudaMemcpy(&Dcutoff, d_far_keys + itarget,
                       sizeof(int), cudaMemcpyDeviceToHost);

            int raw_delta = Dcutoff - Dmin;

            // Log delta values (skip first round)
            if (round_num != 1)
            {
                outfile << raw_delta << "\n";
            }

            // Prevent zero delta
            delta = (raw_delta == 0) ? 1 : raw_delta;
        }
        else
        {
            // Static mode: use predefined delta value
            delta = delta_mode;
            Dcutoff = Dmin + delta;
        }

        last_cutoff = Dcutoff;

        // =====================================================
        // PHASE 1: Light Edge Relaxation (repeat until convergence)
        // =====================================================
        while (true)
        {
            cudaMemset(d_updated, 0, sizeof(int));

            // Relax edges with weight <= delta
            relax_light_edges<<<blocks, threads_per_block>>>(
                d_dist, d_offsets, d_neighs, d_weights,
                N, delta, last_cutoff, d_updated);
            cudaDeviceSynchronize();

            // Check if any distance updates occurred
            int h_updated = 0;
            cudaMemcpy(&h_updated, d_updated, sizeof(int), cudaMemcpyDeviceToHost);

            // Exit loop if no changes (convergence)
            if (h_updated == 0)
                break;
        }

        // =====================================================
        // PHASE 2: Heavy Edge Relaxation (single pass)
        // =====================================================
        cudaMemset(d_updated, 0, sizeof(int));

        // Relax edges with weight > delta (once per round)
        relax_heavy_edges<<<blocks, threads_per_block>>>(
            d_dist, d_offsets, d_neighs, d_weights,
            N, delta, last_cutoff, d_updated);
        cudaDeviceSynchronize();

        // Loop continues: next iteration rebuilds Far queue with updated distances
    }
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

    int *offsets = new int[N + 1]{0};
    for (int i = 0; i <= N; ++i)
    {
        infile >> offsets[i];
    }

    int *neighs = new int[E]{0};
    for (int i = 0; i < E; ++i)
    {
        infile >> neighs[i];
    }

    int *weights = new int[E]{0};
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

    int *h_tent = new int[N]{0}; // sssp distance array. tent means tentative distance
    int *d_dist;
    cudaMalloc(&d_dist, N * sizeof(int));

    // Workspace: allocated ONCE (VRAM allocation policy), reused per source
    int *d_far_vertices, *d_far_keys, *d_updated, *d_count;
    cudaMalloc(&d_far_vertices, N * sizeof(int));
    cudaMalloc(&d_far_keys, N * sizeof(int));
    cudaMalloc(&d_updated, sizeof(int));
    cudaMalloc(&d_count, sizeof(int));

    // Process each source query sequentially
    for (int i = 0; i < S_count; ++i)
    {
        // Get current source vertex
        int source = sources[i];

        // Write source ID to output
        outfile << source << "\n";

        // Run delta-stepping SSSP from this source
        run_delta_stepping_single_source(
            d_dist,
            d_offsets,
            d_neighs,
            d_weights,
            N,
            E,
            source,
            delta_mode,
            K,
            d_far_vertices,
            d_far_keys,
            d_updated,
            d_count,
            outfile);

        // Transfer computed distances from GPU to CPU
        cudaMemcpy(h_tent, d_dist, N * sizeof(int), cudaMemcpyDeviceToHost);

        // Write shortest distances for all vertices
        for (int v = 0; v < N; ++v)
        {
            outfile << h_tent[v] << "\n";
        }
    }

    // Cleanup
    cudaFree(d_offsets);
    cudaFree(d_neighs);
    cudaFree(d_weights);
    cudaFree(d_dist);
    cudaFree(d_far_vertices);
    cudaFree(d_far_keys);
    cudaFree(d_updated);
    cudaFree(d_count);

    delete[] offsets;
    delete[] neighs;
    delete[] weights;
    delete[] h_tent;

    outfile.close();
    return 0;
}
