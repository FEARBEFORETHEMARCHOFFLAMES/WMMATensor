#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_helper.cuh"
#include <cstdlib>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "mma.h";

constexpr int N = 128;
constexpr int K = 128;
constexpr int M = 128;
constexpr int TILE_SIZE = 16;
constexpr int threadsPerSM = 128;
constexpr int blocks = 1;

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/
// -> strg+f wmma

template<typename T> __global__ void matmuladd_simple(T const* const a, T const* const b, T* const c, 
    const int N, const int K, const int M) {
    for (int row = threadIdx.y + blockIdx.y * blockDim.y;
        row < N;
        row += blockDim.y * gridDim.y) {

        for (int col = threadIdx.x + blockIdx.x * blockDim.x;
            col < M;
            col += blockDim.x * gridDim.x) {
            T result = 0;

            for (int k = 0; k < K; k++) {
                result += a[row * N + k] * b[k * M + col];
            }
            c[row * M + col] += result;
        }
    }
}

using namespace nvcuda::wmma;
__global__ void wmma_ker(half* a, half* b, half* c) {
    // declare the fragments
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, half> c_frag;
    // load the inputs
    load_matrix_sync(a_frag, a, 16);
    load_matrix_sync(b_frag, b, 16);
    fill_fragment(c_frag, 0.0f);
    // perform the matrix multiplication
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    // store the output
    store_matrix_sync(c, c_frag, 16, mem_row_major);
}

__global__ void wmma_kernel(half* a, half* b, half* c) {
    // Iterating over the entire matrix in TILE_SIZE x TILE_SIZE chunks
    for (int row = 0; row < M; row += TILE_SIZE) {
        for (int col = 0; col < N; col += TILE_SIZE) {
            // Create fragments for the current tile
            fragment<matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, row_major> a_frag;
            fragment<matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, row_major> b_frag;
            fragment<accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, half> c_frag;

            // Initialize the output fragment to zero
            fill_fragment(c_frag, 0.0f);

            for (int k = 0; k < K; k += TILE_SIZE) {
                // Determine the actual size of the current tile
                int current_tile_height = min(TILE_SIZE, M - row);
                int current_tile_width = min(TILE_SIZE, N - col);
                int current_tile_depth = min(TILE_SIZE, K - k);

                // Load the current tile from matrix A and B into fragments
                load_matrix_sync(a_frag, &a[row * K + k], K); // Adjust the pointer for A
                load_matrix_sync(b_frag, &b[k * N + col], N); // Adjust the pointer for B

                // Perform matrix multiplication for the current tile
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            // Store the result back to matrix C
            store_matrix_sync(&c[row * N + col], c_frag, N, mem_row_major); // Adjust the pointer for C
        }
    }
}

void runMatmulSimple(half* h_a, half* h_b, half* h_c, half* result) {
    half* h_d = new half[N * M];
    half* d_a, * d_b, * d_c;
    CHECK_CUDA(cudaMalloc(&d_a, N * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_b, K * M * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_c, N * M * sizeof(half)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * M * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c, N * M * sizeof(half), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(threadsPerSM);
    dim3 blocksPerGrid(blocks);

    matmuladd_simple<half><<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N, K, M);

    CHECK_CUDA(cudaMemcpy(result, d_c, N * M * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
}

void runWMMA(half* h_a, half* h_b, half* h_c, half* result) {
    half* h_d = new half[N * M];
    half* d_a, * d_b, * d_c;
    CHECK_CUDA(cudaMalloc(&d_a, N * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_b, K * M * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_c, N * M * sizeof(half)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * M * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c, N * M * sizeof(half), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(threadsPerSM);
    dim3 blocksPerGrid(blocks);

    wmma_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);

    CHECK_CUDA(cudaMemcpy(result, d_c, N * M * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
}

int main()
{
    const int dev = 0;
    std::cout << getCUDADeviceInformations(dev).str() << "\n\n";

    half* h_a = new half[N * K]; // N hoch, K breit
    half* h_b = new half[K * M]; // K hoch, M breit
    half* h_c = new half[N * M]; // N hoch, M breit
    std::srand(1337);

    for (size_t i = 0; i < N * K; i++) {
        h_a[i] = std::rand() / ((RAND_MAX + 1u) / 6);  // Note: 1+rand()%6 is biased
    }
    for (size_t i = 0; i < K * M; i++) {
        h_b[i] = std::rand() / ((RAND_MAX + 1u) / 6);  // Note: 1+rand()%6 is biased
    }
    for (size_t i = 0; i < N * M; i++) {
        //It is not possible to load things directly into c in wmma!
        h_c[i] = 0.0f;
    }

    //printMat(h_a, N, K, "h_a");
    //printf("\r\n");
    //printMat(h_b, K, M, "h_b");
    //printf("\r\n");
    //printMat(h_c, N, M, "h_c");
    //printf("\r\n");
    half* simpleResult = new half[N*M];
    runMatmulSimple(h_a, h_b, h_c, simpleResult);
    half* wmmaResult = new half[N*M];
    runWMMA(h_a, h_b, h_c, wmmaResult);
    //printMat(simpleResult, N, M, "simpleResult");
    //printf("\r\n");
    //printMat(wmmaResult, N, M, "wmmaResult");
    //printf("\r\n");

    if (areEqual(simpleResult, wmmaResult, N, M)) {
        printf("Results were equal \r\n");
    }
    else {
        printf("Results were not equal \r\n");
    }
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
