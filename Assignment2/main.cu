#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cuda.h>
using namespace std;

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y BLOCKSIZE_X
#define TILE_WIDTH BLOCKSIZE_X

// __global__ void Mat_mul_1(const int *X, const int *Y, int* Z,
//                         const int p, const int q, const int r){
//     // X is A^T (p x q)
//     // Y is B (q x r)
//     // Z is result (p x r)

//     int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
//     int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

//     if(idx_x >= p || idx_y >= r){
//         return;
//     }

//     for(int i = 0; i < q; i++){
//         Z[idx_x * r + idx_y] += X[i * p + idx_x] * Y[i * r + idx_y];
//     }
// }

// __global__ void Mat_mul_2(const int *X, const int *Y, int* Z,
//                         const int p, const int q, const int r){

//     int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
//     int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

//     if(idx_x >= p || idx_y >= r){
//         return;
//     }

//     for(int i = 0; i < q; i++){
//         Z[idx_x * r + idx_y] += X[idx_x * q + i] * Y[idx_y * q + i];
//     }
// }

__global__ void Mat_mul(const int *A, const int *B, const int *C,
						const int *D, int *E,
						const int p, const int q, const int r)
{

	// Compute E = A^T * B + C * D^T

    // A: q×p, A^T: pxq
    // B: q×r
	// A^T * B : pxq * q×r = p x r

    // C: p×q
    // D: r×q, D^T: q×r
	// C * D^T : pxq * qxr = p x r

    // E is p×r

	__shared__ int tileA[TILE_WIDTH][TILE_WIDTH];
	__shared__ int tileB[TILE_WIDTH][TILE_WIDTH];
	__shared__ int tileC[TILE_WIDTH][TILE_WIDTH];
	__shared__ int tileD[TILE_WIDTH][TILE_WIDTH];

	int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

	int sum = 0;


	// ============ Compute A^T * B ============
    // A^T is p×q, B is q×r, result is p×r
    // For each tile of the q dimension:
	for(int tile = 0; tile < (q + TILE_WIDTH - 1) / TILE_WIDTH; tile++){
        int k = tile * TILE_WIDTH;
        
        // Load A[k+ty][row] into tileA[tx][ty]
        // A is stored q×p, so A[i][j] is at A[i*p + j]
        if(k + ty < q && row < p){
            tileA[tx][ty] = A[(k + ty) * p + row];
        } else {
            tileA[tx][ty] = 0;
        }
        
        // Load B[k+tx][col] into tileB[tx][ty]
        // B is stored q×r, so B[i][j] is at B[i*r + j]
        if(k + tx < q && col < r){
            tileB[tx][ty] = B[(k + tx) * r + col];
        } else {
            tileB[tx][ty] = 0;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for(int i = 0; i < TILE_WIDTH; i++){
            sum += tileA[tx][i] * tileB[i][ty];
        }
        
        __syncthreads();
    }


	// ============ Compute C * D^T ============
    // C is p×q, D is r×q (so D^T is q×r), result is p×r
    // For each tile of the q dimension:
    for(int tile = 0; tile < (q + TILE_WIDTH - 1) / TILE_WIDTH; tile++){
        int k = tile * TILE_WIDTH;
        
        // Load C[row][k+ty] into tileC[tx][ty]
        // C is stored p×q, so C[i][j] is at C[i*q + j]
        if(row < p && k + ty < q){
            tileC[tx][ty] = C[row * q + (k + ty)];
        } else {
            tileC[tx][ty] = 0;
        }
        
        // Load D[col][k+tx] into tileD[tx][ty]
        // D is stored r×q, so D[i][j] is at D[i*q + j]
        if(col < r && k + tx < q){
            tileD[tx][ty] = D[col * q + (k + tx)];
        } else {
            tileD[tx][ty] = 0;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for(int i = 0; i < TILE_WIDTH; i++){
            sum += tileC[tx][i] * tileD[i][ty];
        }
        
        __syncthreads();
    }
    
    // Write result
    if(row < p && col < r){
        E[row * r + col] = sum;
    }

}

// function to compute the output matrix
void compute(int p, int q, int r, int *h_matrixA, int *h_matrixB,
			 int *h_matrixC, int *h_matrixD, int *h_matrixE)
{
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;

	// allocate memory...
	cudaMalloc(&d_matrixA, q * p * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, q * p * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */

	cudaMemset(d_matrixE, 0, p * r * sizeof(int));

	dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y);
	dim3 gridSize((p + BLOCKSIZE_X - 1) / BLOCKSIZE_Y, (r + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);

	Mat_mul<<<gridSize, blockSize>>>(d_matrixA, d_matrixB, d_matrixC, d_matrixD, d_matrixE, p, q, r);

	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			fscanf(inputFilePtr, "%d", &matrix[i * cols + j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			fprintf(outputFilePtr, "%d ", matrix[i * cols + j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv)
{
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;

	inputFilePtr = fopen(inputFileName, "r");
	if (inputFilePtr == NULL)
	{
		printf("Failed to open the input file.!!\n");
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int *)malloc(q * p * sizeof(int));
	matrixB = (int *)malloc(q * r * sizeof(int));
	matrixC = (int *)malloc(p * q * sizeof(int));
	matrixD = (int *)malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, q, p);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int *)malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	compute(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000 * seconds + microSeconds / 1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
