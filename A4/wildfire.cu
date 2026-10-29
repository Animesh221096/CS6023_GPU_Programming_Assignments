// Your solution. This is the only file you submit.
// It is compiled together with the provided harness.cu:
//     nvcc -arch=sm_70 harness.cu wildfire.cu -o main
// See wildfire.h for the definitions you can use and the full contract of
// run_simulation().
#include "wildfire.h"

#define BLOCK_SIZE 256

// ============================================================================
// KERNELS FOR FIRE SIMULATION
// All fire kernels read G (grid at start of step) and write the new grid.
// ============================================================================

// FIX: reads G (snapshot), not the live state grid.
__global__ void compute_heat_kernel(const uint8_t *G, const uint8_t *fuel,
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
            if (G[src_idx] != BURNING)
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

// FIX: fully recomputes the new state from G. A burning-in-G cell loses 1
// fuel and becomes BURNT if it hits 0 (the crew phase may still overwrite
// it with WET afterwards, since wet wins over burnt). Stations are kept
// unburnt defensively.
__global__ void update_fire_kernel(const uint8_t *G, uint8_t *state, uint8_t *fuel,
                                   const int *heat_grid, int N, int I,
                                   const int *stations, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N)
        return;

    int r = idx / N;
    int c = idx % N;
    uint8_t gs = G[idx];

    // Check if this is a fire station (defensive; stations never ignite)
    bool is_station = false;
    for (int i = 0; i < K; i++)
    {
        if (stations[2 * i] == r && stations[2 * i + 1] == c)
        {
            is_station = true;
            break;
        }
    }

    if (gs == BURNING && !is_station)
    {
        // Burning in G: loses 1 fuel; burnt if it runs out.
        // (A crew may still mark this cell WET after this kernel.)
        int nf = (int)fuel[idx] - 1;
        fuel[idx] = (uint8_t)nf;
        state[idx] = (nf == 0) ? BURNT : BURNING;
    }
    else if (gs == UNBURNT && fuel[idx] > 0 && !is_station && heat_grid[idx] >= I)
    {
        // Ignite
        state[idx] = BURNING;
    }
    else
    {
        // Unchanged: stays UNBURT, BURNT, WET, or station cell
        state[idx] = gs;
    }
}

// ============================================================================
// CREW MOVEMENT AND FIREFIGHTING
// FIX: everything is judged from G (grid at the start of the step), and the
// crew writes WET straight into the post-fire state grid, so a cell that
// burned out this step still ends up WET.
// ============================================================================

__device__ int distance_between(int r1, int c1, int r2, int c2, int R)
{
    int dr = abs(r1 - r2);
    int dc = abs(c1 - c2);
    int max_d = (dr > dc) ? dr : dc;
    return (max_d + R - 1) / R; // Ceiling division
}

__device__ void walk_towards(int *r, int *c, int tr, int tc, int R, int N)
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

    // Defensive clamp (cannot overshoot a valid target, but be safe)
    if (*r < 0)
        *r = 0;
    if (*r >= N)
        *r = N - 1;
    if (*c < 0)
        *c = 0;
    if (*c >= N)
        *c = N - 1;
}

// FIX: takes G (start-of-step grid) for all burning checks, and writes WET
// into `state` (the post-fire grid). Multiple crews writing WET to the same
// byte is a benign race (they all write the same value), and each crew
// independently spends 1 water, as the spec requires.
__global__ void move_crews_kernel(int *crews, const uint8_t *G, uint8_t *state,
                                  int N, int C, int R, int X,
                                  const int *stations, int K)
{
    int crew_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (crew_id >= C)
        return;

    int r = crews[3 * crew_id];
    int c = crews[3 * crew_id + 1];
    int w = crews[3 * crew_id + 2];

    // Choose target: judged from G (start of step)
    int closest_r = -1, closest_c = -1, closest_dist = INT_MAX;

    if (w > 0)
    {
        // Closest burning cell, judged from G. A cell burning in G that
        // burns out this very step is STILL a valid target.
        for (int i = 0; i < N * N; i++)
        {
            if (G[i] == BURNING)
            {
                int tr = i / N;
                int tc = i % N;
                int d = distance_between(r, c, tr, tc, R);
                if (d < closest_dist ||
                    (d == closest_dist &&
                     (tr < closest_r || (tr == closest_r && tc < closest_c))))
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
        // Closest fire station
        for (int i = 0; i < K; i++)
        {
            int sr = stations[2 * i];
            int sc = stations[2 * i + 1];
            int d = distance_between(r, c, sr, sc, R);
            if (d < closest_dist ||
                (d == closest_dist &&
                 (sr < closest_r || (sr == closest_r && sc < closest_c))))
            {
                closest_dist = d;
                closest_r = sr;
                closest_c = sc;
            }
        }
    }

    // Move (no move if no target: t == none)
    if (closest_r != -1)
    {
        walk_towards(&r, &c, closest_r, closest_c, R, N);
    }

    // Put out a fire: judged from G. Even if the cell burned out in the fire
    // phase this step, the crew puts it out and it ends the step WET
    // (written over whatever update_fire_kernel left there).
    int cell_idx = r * N + c;
    if (G[cell_idx] == BURNING && w > 0)
    {
        state[cell_idx] = WET;
        w--;
    }

    // Refill: happens whether or not the crew moved (checked after moving)
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
    // d_G is the snapshot of the grid at the START of each step: both the
    // fire phase and the crews read it, so they never see each other's
    // changes within one step.
    uint8_t *d_state, *d_G, *d_fuel, *d_elev;
    int *d_heat, *d_stations, *d_crews;

    cudaMalloc(&d_state, grid_size);
    cudaMalloc(&d_G, grid_size);
    cudaMalloc(&d_fuel, grid_size);
    cudaMalloc(&d_elev, grid_size);
    cudaMalloc(&d_heat, grid_size * sizeof(int));
    if (K > 0)
        cudaMalloc(&d_stations, (size_t)K * 2 * sizeof(int));
    else
        d_stations = nullptr;
    if (C > 0)
        cudaMalloc(&d_crews, (size_t)C * 3 * sizeof(int));
    else
        d_crews = nullptr;

    // Copy to device
    cudaMemcpy(d_state, state, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fuel, fuel, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_elev, elev, grid_size, cudaMemcpyHostToDevice);
    if (K > 0)
        cudaMemcpy(d_stations, stations, (size_t)K * 2 * sizeof(int), cudaMemcpyHostToDevice);
    if (C > 0)
        cudaMemcpy(d_crews, crews, (size_t)C * 3 * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (int)((grid_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Main simulation loop
    for (int step = 0; step < T; step++)
    {
        // --- Snapshot: G = grid at the start of the step ---
        cudaMemcpy(d_G, d_state, grid_size, cudaMemcpyDeviceToDevice);

        // --- Fire phase: reads G, writes d_state/d_fuel ---
        compute_heat_kernel<<<blocks, BLOCK_SIZE>>>(d_G, d_fuel, d_elev, d_heat,
                                                    N, p.H, p.dr, p.dc, p.W, p.S, p.D);
        cudaDeviceSynchronize();

        update_fire_kernel<<<blocks, BLOCK_SIZE>>>(d_G, d_state, d_fuel, d_heat,
                                                   N, p.I, d_stations, K);
        cudaDeviceSynchronize();

        // --- Crew phase: reads G for decisions, writes WET into d_state ---
        if (C > 0)
        {
            int crew_blocks = (C + BLOCK_SIZE - 1) / BLOCK_SIZE;
            move_crews_kernel<<<crew_blocks, BLOCK_SIZE>>>(d_crews, d_G, d_state,
                                                           N, C, p.R, p.X,
                                                           d_stations, K);
            cudaDeviceSynchronize();
        }
        // No combine copy needed: crews already wrote WET into the post-fire
        // grid, so wet wins over burnt automatically.
    }

    // Copy results back
    cudaMemcpy(state, d_state, grid_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(fuel, d_fuel, grid_size, cudaMemcpyDeviceToHost);
    if (C > 0)
        cudaMemcpy(crews, d_crews, (size_t)C * 3 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_state);
    cudaFree(d_G);
    cudaFree(d_fuel);
    cudaFree(d_elev);
    cudaFree(d_heat);
    if (d_stations)
        cudaFree(d_stations);
    if (d_crews)
        cudaFree(d_crews);
}
