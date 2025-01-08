#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_helper.cuh"
#include <cstdlib>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "mma.h";

constexpr auto N = 4;
constexpr auto K = 4;
constexpr auto M = 4;

constexpr auto threadsPerSM = 128;

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

//using namespace nvcuda::wmma;
//__global__ void wmma_ker(half* a, half* b, float* c) {
//    // Declare the fragments
//    fragment<matrix_a, 16, 16, 16, half, col_major> a_frag;
//    fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
//    fragment<accumulator, 16, 16, 16, float> c_frag;
//    // Initialize the output to zero
//    fill_fragment(c_frag, 0.0f);
//    // Load the inputs
//    load_matrix_sync(a_frag, a, 16);
//    load_matrix_sync(b_frag, b, 16);
//    // Perform the matrix multiplication
//    mma_sync(c_frag, a_frag, b_frag, c_frag);
//    // Store the output
//    store_matrix_sync(c, c_frag, 16, mem_row_major);
//}

int main()
{
    const int dev = 0;
    std::cout << getCUDADeviceInformations(dev).str() << "\n\n";

    half h_a[N * K]; // N hoch, K breit
    half h_b[K * M]; // K hoch, M breit
    half h_c[N * M]; // N hoch, M breit
    std::srand(1337);

    for (size_t i = 0; i < N * K; i++) {
        h_a[i] = std::rand() / ((RAND_MAX + 1u) / 6);  // Note: 1+rand()%6 is biased
    }
    for (size_t i = 0; i < K * M; i++) {
        h_b[i] = std::rand() / ((RAND_MAX + 1u) / 6);  // Note: 1+rand()%6 is biased
    }
    for (size_t i = 0; i < N * M; i++) {
        h_c[i] = std::rand() / ((RAND_MAX + 1u) / 6);  // Note: 1+rand()%6 is biased
    }
    
    half* d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, N * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_b, K * M * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_c, N * M * sizeof(half)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * M * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c, N * M * sizeof(half), cudaMemcpyHostToDevice));

    printMat(h_a, N, K, "h_a");
    printf("\r\n");
    printMat(h_b, K, M, "h_b");
    printf("\r\n");
    printMat(h_c, N, M, "h_c");
    printf("\r\n");

    dim3 threadsPerBlock(128);
    dim3 blocksPerGrid(16,1);

    matmuladd_simple<half> <<<blocksPerGrid, threadsPerBlock >>> (d_a, d_b, d_c, N, K, M);

    CHECK_CUDA(cudaMemcpy(h_c, d_c, N * M * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
    printMat(h_c, N, M, "h_c");
    printf("\r\n");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
