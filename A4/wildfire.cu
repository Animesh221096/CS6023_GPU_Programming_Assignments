// Your solution. This is the only file you submit.
// It is compiled together with the provided harness.cu:
//     nvcc -arch=sm_70 harness.cu wildfire.cu -o main
// See wildfire.h for the definitions you can use and the full contract of
// run_simulation().
#include "wildfire.h"

#define BLOCK_SIZE 256 // you can change it.
#define MAX_BURNING 1000000

// ============================================================================
// CUDA KERNELS
// ============================================================================

// ============================================================================
// KERNELS FOR FIRE SIMULATION
// ============================================================================

__global__ void compute_heat_kernel(const uint8_t *state, const uint8_t *fuel,
                                    const uint8_t *elev, int *heat_grid,
                                    int N, int H, int dr, int dc, int W, int S, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N)
        return;

    int r = idx / N;
    int c = idx % N;
    int heat = 0;

    // Check all cells in 5x5 neighbourhood
    for (int rs = r - 2; rs <= r + 2; rs++)
    {
        for (int cs = c - 2; cs <= c + 2; cs++)
        {
            if (rs < 0 || rs >= N || cs < 0 || cs >= N)
                continue;
            if (rs == r && cs == c)
                continue;

            int src_idx = rs * N + cs;
            if (state[src_idx] != BURNING)
                continue;

            int a = H;
            int b = 1;

            // Wind effect
            int or_ = r - rs;
            int oc = c - cs;
            if ((or_ == dr && oc == dc) || (or_ == 2 * dr && oc == 2 * dc))
            {
                a = a * W;
            }
            else if ((or_ == -dr && oc == -dc) || (or_ == -2 * dr && oc == -2 * dc))
            {
                b = b * W;
            }

            // Slope effect
            int delta = (int)elev[idx] - (int)elev[src_idx];
            if (delta > D)
            {
                a = a * S;
            }
            else if (delta < -D)
            {
                b = b * S;
            }

            heat += (a / b);
        }
    }

    heat_grid[idx] = heat;
}

__global__ void update_fire_kernel(uint8_t *state, uint8_t *fuel, const int *heat_grid,
                                   int N, int I, const int *stations, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N)
        return;

    int r = idx / N;
    int c = idx % N;

    // Check if this is a fire station
    bool is_station = false;
    for (int i = 0; i < K; i++)
    {
        if (stations[2 * i] == r && stations[2 * i + 1] == c)
        {
            is_station = true;
            break;
        }
    }

    // Ignite unburnt cells
    if (state[idx] == UNBURNT && fuel[idx] > 0 && !is_station && heat_grid[idx] >= I)
    {
        state[idx] = BURNING;
    }

    // Burn existing burning cells
    if (state[idx] == BURNING)
    {
        fuel[idx]--;
        if (fuel[idx] == 0)
        {
            state[idx] = BURNT;
        }
    }
}

// ============================================================================
// CREW MOVEMENT AND FIREFIGHTING
// ============================================================================

__device__ int distance_between(int r1, int c1, int r2, int c2, int R)
{
    int dr = abs(r1 - r2);
    int dc = abs(c1 - c2);
    int max_d = (dr > dc) ? dr : dc;
    return (max_d + R - 1) / R; // Ceiling division
}

__device__ void walk_towards(int *r, int *c, int tr, int tc, int R)
{
    int dr = tr - *r;
    int dc = tc - *c;

    if (dr > 0)
        *r += (R < dr) ? R : dr;
    else if (dr < 0)
        *r -= (R < -dr) ? R : -dr;

    if (dc > 0)
        *c += (R < dc) ? R : dc;
    else if (dc < 0)
        *c -= (R < -dc) ? R : -dc;
}

__global__ void move_crews_kernel(int *crews, const uint8_t *state, uint8_t *new_state,
                                  int N, int C, int R, int X,
                                  const int *stations, int K)
{
    int crew_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (crew_id >= C)
        return;

    int r = crews[3 * crew_id];
    int c = crews[3 * crew_id + 1];
    int w = crews[3 * crew_id + 2];

    // Find closest burning cell
    int closest_r = -1, closest_c = -1, closest_dist = INT_MAX;

    if (w > 0)
    {
        for (int i = 0; i < N * N; i++)
        {
            if (state[i] == BURNING)
            {
                int tr = i / N;
                int tc = i % N;
                int d = distance_between(r, c, tr, tc, R);
                if (d < closest_dist || (d == closest_dist && (tr < closest_r || (tr == closest_r && tc < closest_c))))
                {
                    closest_dist = d;
                    closest_r = tr;
                    closest_c = tc;
                }
            }
        }
    }
    else
    {
        // Find closest fire station
        for (int i = 0; i < K; i++)
        {
            int sr = stations[2 * i];
            int sc = stations[2 * i + 1];
            int d = distance_between(r, c, sr, sc, R);
            if (d < closest_dist || (d == closest_dist && (sr < closest_r || (sr == closest_r && sc < closest_c))))
            {
                closest_dist = d;
                closest_r = sr;
                closest_c = sc;
            }
        }
    }

    // Move towards target
    if (closest_r != -1)
    {
        walk_towards(&r, &c, closest_r, closest_c, R);
    }

    // Put out fire or refill
    int cell_idx = r * N + c;
    if (state[cell_idx] == BURNING && w > 0)
    {
        new_state[cell_idx] = WET;
        w--;
    }

    // Check if on fire station
    for (int i = 0; i < K; i++)
    {
        if (stations[2 * i] == r && stations[2 * i + 1] == c)
        {
            w = X;
            break;
        }
    }

    crews[3 * crew_id] = r;
    crews[3 * crew_id + 1] = c;
    crews[3 * crew_id + 2] = w;
}

// ============================================================================
// SIMULATION DRIVER
// ============================================================================

void run_simulation(Params p, uint8_t *state, uint8_t *fuel, const uint8_t *elev,
                    int F, const int *fires, int K, const int *stations,
                    int C, int *crews)
{
    int N = p.N;
    int T = p.T;
    size_t grid_size = (size_t)N * N;

    // Initialize fires
    for (int i = 0; i < F; i++)
    {
        int r = fires[2 * i];
        int c = fires[2 * i + 1];
        state[r * N + c] = BURNING;
    }

    // Allocate device memory
    uint8_t *d_state, *d_fuel, *d_elev, *d_new_state;
    int *d_heat, *d_stations, *d_crews;

    cudaMalloc(&d_state, grid_size);
    cudaMalloc(&d_fuel, grid_size);
    cudaMalloc(&d_elev, grid_size);
    cudaMalloc(&d_new_state, grid_size);
    cudaMalloc(&d_heat, grid_size * sizeof(int));
    cudaMalloc(&d_stations, K * 2 * sizeof(int));
    cudaMalloc(&d_crews, C * 3 * sizeof(int));

    // Copy to device
    cudaMemcpy(d_state, state, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fuel, fuel, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_elev, elev, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_stations, stations, K * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_crews, crews, C * 3 * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (N * N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Main simulation loop
    for (int step = 0; step < T; step++)
    {
        // Fire phase
        compute_heat_kernel<<<blocks, BLOCK_SIZE>>>(d_state, d_fuel, d_elev, d_heat,
                                                    N, p.H, p.dr, p.dc, p.W, p.S, p.D);
        cudaDeviceSynchronize();

        update_fire_kernel<<<blocks, BLOCK_SIZE>>>(d_state, d_fuel, d_heat, N, p.I,
                                                   d_stations, K);
        cudaDeviceSynchronize();

        // Crew phase
        cudaMemcpy(d_new_state, d_state, grid_size, cudaMemcpyDeviceToDevice);

        int crew_blocks = (C + BLOCK_SIZE - 1) / BLOCK_SIZE;
        move_crews_kernel<<<crew_blocks, BLOCK_SIZE>>>(d_crews, d_state, d_new_state,
                                                       N, C, p.R, p.X, d_stations, K);
        cudaDeviceSynchronize();

        // Combine phase (wet wins over burnt)
        cudaMemcpy(d_state, d_new_state, grid_size, cudaMemcpyDeviceToDevice);
    }

    // Copy results back
    cudaMemcpy(state, d_state, grid_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(fuel, d_fuel, grid_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(crews, d_crews, C * 3 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_state);
    cudaFree(d_fuel);
    cudaFree(d_elev);
    cudaFree(d_new_state);
    cudaFree(d_heat);
    cudaFree(d_stations);
    cudaFree(d_crews);
}
