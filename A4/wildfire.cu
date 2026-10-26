// Your solution. This is the only file you submit.
// It is compiled together with the provided harness.cu:
//     nvcc -arch=sm_70 harness.cu wildfire.cu -o main
// See wildfire.h for the definitions you can use and the full contract of
// run_simulation().
#include "wildfire.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <climits>
#include <algorithm>

#define BLOCK_SIZE 256 // you can change it.

// ============================================================================
// CUDA KERNELS
// ============================================================================


__device__ inline int idx(int r, int c, int N) { return r * N + c; }
__device__ inline int abs_i(int x) { return x < 0 ? -x : x; }
__device__ inline int max_i(int a, int b) { return a > b ? a : b; }
__device__ inline int min_i(int a, int b) { return a < b ? a : b; }

// Set initial fires (called once at start)
__global__ void init_fires_kernel(int F, const int *fires, int N, uint8_t *state)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < F)
    {
        int r = fires[2 * i], c = fires[2 * i + 1];
        state[idx(r, c, N)] = BURNING;
    }
}

// Fire spread: each thread handles one cell
__global__ void fire_kernel(int N, int H, int dr, int dc, int W, int S, int D, int I,
                            const uint8_t *old_state, const uint8_t *fuel,
                            const uint8_t *elev, const uint8_t *station_map,
                            uint8_t *new_state, uint8_t *new_fuel)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * N)
        return;

    int r = tid / N, c = tid % N;
    uint8_t st = old_state[tid];
    uint8_t fu = fuel[tid];
    uint8_t el = elev[tid];
    bool sta = (station_map[tid] == 1);

    // Compute heat from all burning neighbors in 5x5 window
    int heat = 0;
    for (int dsr = -2; dsr <= 2; dsr++)
    {
        for (int dsc = -2; dsc <= 2; dsc++)
        {
            if (dsr == 0 && dsc == 0)
                continue;
            int sr = r + dsr, sc = c + dsc;
            if (sr < 0 || sr >= N || sc < 0 || sc >= N)
                continue;
            int sid = idx(sr, sc, N);
            if (old_state[sid] != BURNING)
                continue;

            int or_ = -dsr, oc_ = -dsc; // relative pos from source
            int a = H, b = 1;

            // Wind effect
            if ((or_ == dr && oc_ == dc) || (or_ == 2 * dr && oc_ == 2 * dc))
                a *= W;
            else if ((or_ == -dr && oc_ == -dc) || (or_ == -2 * dr && oc_ == -2 * dc))
                b *= W;

            // Slope effect
            int delta = el - elev[sid];
            if (delta > D)
                a *= S;
            else if (delta < -D)
                b *= S;

            heat += a / b;
        }
    }

    // Update state and fuel
    uint8_t nst = st, nfu = fu;
    if (st == BURNING)
    {
        nfu = max_i(0, (int)fu - 1);
        if (nfu == 0)
            nst = BURNT;
    }
    else if (st == UNBURNT && !sta && fu > 0 && heat >= I)
    {
        nst = BURNING;
    }

    new_state[tid] = nst;
    new_fuel[tid] = nfu;
}

// Each crew finds closest target and moves
__global__ void crew_kernel(int N, int R, int X, int C, int *crews,
                            const uint8_t *old_state, const int *stations, int K,
                            const int *burning_list, int num_burning,
                            uint8_t *extinguish)
{
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= C)
        return;

    int off = cid * 3;
    int cr = crews[off], cc = crews[off + 1], cw = crews[off + 2];

    int tr = -1, tc = -1, bdist = INT_MAX;

    if (cw > 0)
    {
        // Find closest burning cell
        if (num_burning > 0)
        {
            for (int i = 0; i < num_burning; i++)
            {
                int br = burning_list[2 * i], bc = burning_list[2 * i + 1];
                int d = max_i(abs_i(br - cr), abs_i(bc - cc));
                int steps = (d + R - 1) / R; // ceil division
                if (steps < bdist || (steps == bdist && (br < tr || (br == tr && bc < tc))))
                {
                    bdist = steps;
                    tr = br;
                    tc = bc;
                }
            }
        }
    }
    else if (cw == 0 && K > 0)
    {
        // Find closest station
        for (int i = 0; i < K; i++)
        {
            int sr = stations[2 * i], sc = stations[2 * i + 1];
            int d = max_i(abs_i(sr - cr), abs_i(sc - cc));
            int steps = (d + R - 1) / R;
            if (steps < bdist || (steps == bdist && (sr < tr || (sr == tr && sc < tc))))
            {
                bdist = steps;
                tr = sr;
                tc = sc;
            }
        }
    }

    // Move towards target
    int nr = cr, nc = cc;
    if (tr >= 0)
    {
        int dr_ = tr - cr, dc_ = tc - cc;
        int mr = dr_ > 0 ? min_i(R, dr_) : max_i(-R, dr_);
        int mc = dc_ > 0 ? min_i(R, dc_) : max_i(-R, dc_);
        nr = cr + mr;
        nc = cc + mc;
    }

    // Check if at fire station
    bool is_sta = false;
    for (int i = 0; i < K; i++)
    {
        if (stations[2 * i] == nr && stations[2 * i + 1] == nc)
        {
            is_sta = true;
            break;
        }
    }

    // Put out fire or refill
    int nw = cw;
    int nidx = nr * N + nc;
    if (old_state[nidx] == BURNING && cw > 0)
    {
        extinguish[nidx] = 1;
        nw -= 1;
    }
    if (is_sta)
        nw = X;

    crews[off] = nr;
    crews[off + 1] = nc;
    crews[off + 2] = nw;
}


// ============================================================================
// SIMULATION DRIVER
// ============================================================================

void run_simulation(Params p, uint8_t *state, uint8_t *fuel, const uint8_t *elev,
                    int F, const int *fires, int K, const int *stations,
                    int C, int *crews)
{
    int N = p.N, cells = N * N;
    size_t cells_bytes = (size_t)cells;
    size_t crews_bytes = (size_t)C * 3 * sizeof(int);

    // Device memory allocation using Thrust - use uint8_t instead of bool
    thrust::device_vector<uint8_t> d_state(state, state + cells);
    thrust::device_vector<uint8_t> d_new_state(cells);
    thrust::device_vector<uint8_t> d_fuel(fuel, fuel + cells);
    thrust::device_vector<uint8_t> d_new_fuel(cells);
    thrust::device_vector<uint8_t> d_elev(elev, elev + cells);
    thrust::device_vector<uint8_t> d_station(cells, 0); // Changed from bool to uint8_t
    thrust::device_vector<uint8_t> d_extinguish(cells, 0);
    thrust::device_vector<int> d_fires(F * 2);
    thrust::device_vector<int> d_stations(K * 2);
    thrust::device_vector<int> d_crews(C * 3);

    // Copy input data to device
    if (F > 0)
    {
        cudaMemcpy(d_fires.data().get(), fires, F * 2 * sizeof(int), cudaMemcpyHostToDevice);
    }
    if (K > 0)
    {
        cudaMemcpy(d_stations.data().get(), stations, K * 2 * sizeof(int), cudaMemcpyHostToDevice);
    }
    if (C > 0)
    {
        cudaMemcpy(d_crews.data().get(), crews, crews_bytes, cudaMemcpyHostToDevice);
    }

    // Build station map
    thrust::host_vector<uint8_t> h_station(cells, 0);
    for (int i = 0; i < K; i++)
    {
        int r = stations[2 * i], c = stations[2 * i + 1];
        h_station[r * N + c] = 1;
    }
    cudaMemcpy(thrust::raw_pointer_cast(d_station.data()), h_station.data(), cells_bytes, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_station.data().get(), h_station.data().get(), cells_bytes, cudaMemcpyHostToDevice);

    // Set initial fires
    dim3 block(BLOCK_SIZE);
    dim3 grid((F + BLOCK_SIZE - 1) / BLOCK_SIZE);
    init_fires_kernel<<<grid, block>>>(F, d_fires.data().get(), N, d_state.data().get());
    cudaDeviceSynchronize();

    // Main simulation loop
    for (int step = 0; step < p.T; step++)
    {
        // Collect burning cells (host scan - can be optimized later)
        thrust::host_vector<uint8_t> h_state = d_state;
        thrust::host_vector<int> h_burn(2 * cells);
        int nb = 0;
        for (int i = 0; i < cells; i++)
        {
            if (h_state[i] == BURNING)
            {
                h_burn[2 * nb] = i / N;
                h_burn[2 * nb + 1] = i % N;
                nb++;
            }
        }
        thrust::device_vector<int> d_burn(2 * nb);
        if (nb > 0)
        {
            cudaMemcpy(thrust::raw_pointer_cast(d_burn.data()), h_burn.data(), 2 * nb * sizeof(int), cudaMemcpyHostToDevice);
        }
        // if (nb > 0) cudaMemcpy(d_burn.data().get(), h_burn.data().get(), 2*nb*sizeof(int), cudaMemcpyHostToDevice);

        // Reset extinguish flags (FIXED: use bytes, use .data().get())
        cudaMemset(d_extinguish.data().get(), 0, cells_bytes);

        // Fire kernel: compute heat and update states
        grid = dim3((cells + BLOCK_SIZE - 1) / BLOCK_SIZE);
        fire_kernel<<<grid, block>>>(
            N, p.H, p.dr, p.dc, p.W, p.S, p.D, p.I,
            d_state.data().get(), d_fuel.data().get(), d_elev.data().get(),
            d_station.data().get(), d_new_state.data().get(), d_new_fuel.data().get());

        // Crew kernel: movement and extinguishing
        if (C > 0)
        {
            dim3 cblock(min(C, BLOCK_SIZE));
            dim3 cgrid((C + cblock.x - 1) / cblock.x);
            crew_kernel<<<cgrid, cblock>>>(
                N, p.R, p.X, C, d_crews.data().get(),
                d_state.data().get(), d_stations.data().get(), K,
                nb > 0 ? d_burn.data().get() : nullptr, nb,
                d_extinguish.data().get());
        }

        cudaDeviceSynchronize();

        // Apply extinguish flags (wet overrides burnt)
        thrust::host_vector<uint8_t> h_new = d_new_state;
        thrust::host_vector<uint8_t> h_ex = d_extinguish;
        for (int i = 0; i < cells; i++)
        {
            if (h_ex[i] == 1)
                h_new[i] = WET;
        }
        cudaMemcpy(thrust::raw_pointer_cast(d_state.data()), h_new.data(), cells_bytes, cudaMemcpyHostToDevice);
        // cudaMemcpy(d_state.data().get(), h_new.data().get(), cells_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_fuel.data().get(), d_new_fuel.data().get(), cells_bytes, cudaMemcpyDeviceToHost);
    }

    // Copy results back to host
    cudaMemcpy(state, d_state.data().get(), cells_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(fuel, d_fuel.data().get(), cells_bytes, cudaMemcpyDeviceToHost);
    if (C > 0)
    {
        cudaMemcpy(crews, d_crews.data().get(), crews_bytes, cudaMemcpyDeviceToHost);
    }
}
