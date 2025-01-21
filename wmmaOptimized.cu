#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_helper.cuh"
#include <cstdlib>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "mma.h";
#include <fstream>

constexpr int TILE_SIZE = 16;
constexpr int WARP_SIZE = 32;

// WMMA guide/article: https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/#programmatic_access_to_tensor_cores_in_cuda_90
// Nvidia programming guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-matrix-functions
// Float precision: https://blog.demofox.org/2017/11/21/floating-point-precision/
// Double precision in tensors: https://blogs.nvidia.com/blog/double-precision-tensor-cores/ -> in hpc alpha?
// Which fragment sizes are supported?: https://forums.developer.nvidia.com/t/why-does-wmma-and-mma-support-different-matrix-tile-size/271067 -> https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-wmma-mma
// Blocks are only launched if all of it's warps can be resident: https://stackoverflow.com/questions/6605581/what-is-the-context-switching-mechanism-in-gpu
// Number of blocks and threads per block should constitute a "full load", i.e. maximum resident threads, to yield good occupancy: https://forums.developer.nvidia.com/t/question-about-threads-per-block-and-warps-per-sm/77491
// Very good explanation of warp and block scheduling: https://stackoverflow.com/questions/64624793/warp-and-block-scheduling-in-cuda-what-exactly-happens-and-questions-about-el
// Choosing number of blocks and threads: https://forums.developer.nvidia.com/t/how-to-choose-how-many-threads-blocks-to-have/55529
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
                result += a[row * K + k] * b[k * M + col];
            }
            c[row * M + col] += result;
        }
    }
}

using namespace nvcuda::wmma;
//Perform tiled matrix matrix multiplication
//WMMA works on tile sizes of 16
//We divide the matrix into equal sized tiles of 16
//An output tile in the matrix c is the result of C_ij = Sum over k(A_ik * B_kj)
// => basically, it works just like the element wise multiplication. Except that every element is replaced with a 16x16 tile!#
// If the input matrices do not fit the tiling neatly, we will have to manually calculate the remainder matrix parts
// -> this is probably
template<typename T> __global__ void wmmaFixedAmountOfThreads(T* a, T* b, T* c, T* a_temp, T* b_temp, const int N, const int K, const int M) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / WARP_SIZE;
    int tileIdX = warpId % (M / TILE_SIZE);
    int tileIdY = warpId / (M / TILE_SIZE);

    /*if (tileIdY >= N) {
        printf("Error, tileIdY was greater than N");
        return;
    }
    printf("(%d, %d)", tileIdX, tileIdY);*/
    fragment<matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, T, row_major> a_frag;
    fragment<matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, T, row_major> b_frag;
    fragment<accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, T> c_frag;
    bool printedOnce = false;
    fill_fragment(c_frag, 0.0f);
    for (int k = 0; k < K; k += TILE_SIZE) {
        // The last argument is the stride between consecutive rows -> i.e. we load 16 elements from the first row, how many elements to skip to get to the next row?
        // -> amount of columns, K for a and N for b
        // We have to start the load at a 256bit aligned position (16x16=256), do pointer arithmetic to figure out start of tile
        load_matrix_sync(a_frag, &a[tileIdY * TILE_SIZE * K + k], K);
        load_matrix_sync(b_frag, &b[k * M + tileIdX * TILE_SIZE], M);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    // store result of tile sum(c_frag) in the corresponding c tile. N is again the number of columns(stride between rows of the tile)
    store_matrix_sync(&c[tileIdY * TILE_SIZE * M + tileIdX * TILE_SIZE], c_frag, M, mem_row_major);
}

template<typename T> __global__ void wmmaGridStride(T* a, T* b, T* c, T* a_temp, T* b_temp, const int N, const int K, const int M) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / WARP_SIZE;
    int tileIdX = warpId % (M / TILE_SIZE);
    int tileIdY = warpId / (M / TILE_SIZE);
    bool firstIteration = true;
    while(tileIdY < (N / TILE_SIZE)) {
        /*if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("%d, %d\r\n", tileIdX, tileIdY);
        }*/
        fragment<matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, T, row_major> a_frag;
        fragment<matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, T, row_major> b_frag;
        fragment<accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, T> c_frag;
        fill_fragment(c_frag, 0.0f);
        for (int k = 0; k < K; k += TILE_SIZE) {
            // The last argument is the stride between consecutive rows -> i.e. we load 16 elements from the first row, how many elements to skip to get to the next row?
            // -> amount of columns, K for a and N for b
            // We have to start the load at a 256bit aligned position (16x16=256), do pointer arithmetic to figure out start of tile
            load_matrix_sync(a_frag, &a[tileIdY * TILE_SIZE * K + k], K);
            load_matrix_sync(b_frag, &b[k * M + tileIdX * TILE_SIZE], M);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        // store result of tile sum(c_frag) in the corresponding c tile. N is again the number of columns(stride between rows of the tile)
        store_matrix_sync(&c[tileIdY * TILE_SIZE * M + tileIdX * TILE_SIZE], c_frag, M, mem_row_major);
        
        firstIteration = false;
        threadId += blockDim.x * gridDim.x;
        warpId = threadId / WARP_SIZE;
        tileIdX = warpId % (M / TILE_SIZE);
        tileIdY = warpId / (M / TILE_SIZE);
    }
}

template<typename T> void runMatmulSimple(T* h_a, T* h_b, T* h_c, T* result, const int N, const int K, const int M, const int threadsPerBlock, const int blocks) {
    T* h_d = new T[N * M];
    T* d_a, * d_b, * d_c;
    CHECK_CUDA(cudaMalloc(&d_a, N * K * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_b, K * M * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_c, N * M * sizeof(T)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * K * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * M * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c, N * M * sizeof(T), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock3D(threadsPerBlock);
    dim3 blocksPerGrid3D(blocks);

    matmuladd_simple<T><<<blocksPerGrid3D, threadsPerBlock3D>>>(d_a, d_b, d_c, N, K, M);

    CHECK_CUDA(cudaMemcpy(result, d_c, N * M * sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
}

template<typename T> void runWMMAFixedAmountOfThreads(T* h_a, T* h_b, T* h_c, T* result, const int N, const int K, const int M, const int threadsPerBlock, const int blocks) {
    T* h_d = new T[N * M];
    T* d_a, * d_b, * d_c;
    T* d_a_temp, *d_b_temp;
    CHECK_CUDA(cudaMalloc(&d_a, N * K * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_b, K * M * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_c, N * M * sizeof(T)));

    CHECK_CUDA(cudaMalloc(&d_a_temp, TILE_SIZE * TILE_SIZE * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_b_temp, TILE_SIZE * TILE_SIZE * sizeof(T)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * K * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * M * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c, N * M * sizeof(T), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock3D(threadsPerBlock);
    dim3 blocksPerGrid3D(blocks);

    wmmaFixedAmountOfThreads<T><<<blocksPerGrid3D, threadsPerBlock3D>>>(d_a, d_b, d_c, d_a_temp, d_b_temp, N, K, M);

    CHECK_CUDA(cudaMemcpy(result, d_c, N * M * sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
}

template<typename T> void runWMMAGridStride(T* h_a, T* h_b, T* h_c, T* result, const int N, const int K, const int M, const int threadsPerBlock, const int blocks) {
    T* h_d = new T[N * M];
    T* d_a, * d_b, * d_c;
    T* d_a_temp, * d_b_temp;
    CHECK_CUDA(cudaMalloc(&d_a, N * K * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_b, K * M * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_c, N * M * sizeof(T)));

    CHECK_CUDA(cudaMalloc(&d_a_temp, TILE_SIZE * TILE_SIZE * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_b_temp, TILE_SIZE * TILE_SIZE * sizeof(T)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * K * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * M * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c, N * M * sizeof(T), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock3D(threadsPerBlock);
    dim3 blocksPerGrid3D(blocks);

    wmmaGridStride<T> << <blocksPerGrid3D, threadsPerBlock3D>> > (d_a, d_b, d_c, d_a_temp, d_b_temp, N, K, M);

    CHECK_CUDA(cudaMemcpy(result, d_c, N * M * sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
}

void createTestFile() {
    const int dev = 0;
    int threadsPerSM = 128;
    int blocks = 1;
    std::cout << getCUDADeviceInformations(dev).str() << "\n\n";
    std::srand(1337);

    int sizes[5] = { 256, 512, 1024, 2048, 4096 };
    int repetitions = 10;
    int N, K, M = 0;
    std::ofstream myfile;
    half divisor_half = (half)(RAND_MAX + 1u);
    double divisor_double = (double)(RAND_MAX + 1u);
    myfile.open("DoubleSimple_vs_HalfWMMA_1_2.csv");
    for (int s = 0; s < 5; s++) {
        N = sizes[s];
        M = sizes[s];
        K = sizes[s];
        threadsPerSM = std::min(256, sizes[s] / 2);
        blocks = sizes[s] / threadsPerSM / 2;
        half* h_a = new half[N * K]; // N hoch, K breit
        double* h_a_2 = new double[N * K]; // N hoch, K breit
        half* h_b = new half[K * M]; // K hoch, M breit
        double* h_b_2 = new double[K * M]; // K hoch, M breit
        half* h_c = new half[N * M]; // N hoch, M breit
        double* h_c_2 = new double[N * M]; // N hoch, M breit
        for (int i = 0; i < repetitions; i++) {
            for (size_t i = 0; i < N * K; i++) {
                int randomNumber = std::rand();
                h_a[i] = (half)1.0f + (half)randomNumber / divisor_half;
                h_a_2[i] = 1.0f + (double)randomNumber / divisor_double;
            }
            for (size_t i = 0; i < K * M; i++) {
                int randomNumber = std::rand();
                h_b[i] = (half)1.0f + (half)randomNumber / divisor_half;
                h_b_2[i] = 1.0f + (double)randomNumber / divisor_double;
            }
            for (size_t i = 0; i < N * M; i++) {
                //It is not possible to load things directly into c in wmma!
                h_c[i] = 0.0f;
                h_c_2[i] = 0.0f;
            }
            std::cout << "Done with generating data" << std::endl;
            half* result1 = new half[N * M];
            runWMMAFixedAmountOfThreads<half>(h_a, h_b, h_c, result1, N, K, M, threadsPerSM, blocks);
            std::cout << "Done with half simple" << std::endl;
            double* result2 = new double[N * M];
            runMatmulSimple<double>(h_a_2, h_b_2, h_c_2, result2, N, K, M, threadsPerSM, blocks);
            std::cout << "Done with float simple" << std::endl;

            long double totalDiff = totalDifference(result2, result1, N, M);
            std::cout << "Calculation done for size " << sizes[s] << " and repetition " << i << std::endl;
            std::cout << "Total difference between regular and WMMA result: " << totalDiff << std::endl;
            std::cout << "Average difference between regular and WMMA result: " << totalDiff / (long double)(N * M) << std::endl;
            std::cout << "The range of input numbers was half numbers in [0,6] " << std::endl << std::endl;
            myfile << totalDiff << ";";
        }
        myfile << "\r\n";
    }
    myfile.close();
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
}

int main()
{
    const int dev = 0;
    std::cout << getCUDADeviceInformations(dev).str() << "\n\n";
    int M = 1024, K = 1024, N = 1024;
    // the number of threads should be 32 * (N/16) * (M/16)
    // adjust blocks such that this holds
    int totalThreads = 32 * (N / 16) * (M / 16);
    int threadsPerBlock = 128;
    int blocks = totalThreads / threadsPerBlock;
    
    half* h_a = new half[N * K]; // N hoch, K breit
    float* h_a_2 = new float[N * K]; // N hoch, K breit
    half* h_b = new half[K * M]; // K hoch, M breit
    float* h_b_2 = new float[K * M]; // K hoch, M breit
    half* h_c = new half[N * M]; // N hoch, M breit
    float* h_c_2 = new float[N * M]; // N hoch, M breit

    half divisor_half = (half)(RAND_MAX + 1u);
    float divisor_double = (float)(RAND_MAX + 1u);

    for (size_t i = 0; i < N * K; i++) {
        int randomNumber = std::rand();
        h_a[i] = (half)randomNumber / divisor_half;
        h_a_2[i] = (float)randomNumber / divisor_double;
    }

    for (size_t i = 0; i < K * M; i++) {    
        int randomNumber = std::rand();
        h_b[i] = (half)randomNumber / divisor_half;
        h_b_2[i] = (float)randomNumber / divisor_double;
    }

    /*std::cout << "h_a: [ ";
    for (int i = 0; i < N; i++) {
        std::cout << __half2float(h_a[i * N + K - 1]) << " ";
    }*/
    //std::cout << "]\r\n";

    /*std::cout << "h_b: [ ";
    for (int i = 0; i < N; i++) {
        std::cout << __half2float(h_b[(K-1) * M + i]) << " ";
    }
    std::cout << "]\r\n";*/
    for (size_t i = 0; i < N * M; i++) {
        //It is not possible to load things directly into c in wmma!
        h_c[i] = 0.0f;
        h_c_2[i] = 0.0f;
    }
    half* result1 = new half[N * M];
    TimerCPU timerCPU;
    timerCPU.startTimer();
    runWMMAFixedAmountOfThreads<half>(h_a, h_b, h_c, result1, N, K, M, threadsPerBlock, blocks);
    std::cout << "WMMA kernel before took " << timerCPU.stopTimer() << " milliseconds\r\n";

    half* result2 = new half[N * M];
    TimerCPU timerCPU2;
    timerCPU2.startTimer();
    runWMMAGridStride<half>(h_a, h_b, h_c, result2, N, K, M, threadsPerBlock, blocks / 64);
    std::cout << "WMMA kernel after took " << timerCPU2.stopTimer() << " milliseconds\r\n";

   // printMat(result2, N, M, "result2");

    float* result3 = new float[N * M];
    TimerCPU timerCPU3;
    timerCPU3.startTimer();
    runMatmulSimple<float>(h_a_2, h_b_2, h_c_2, result3, N, K, M, threadsPerBlock, blocks);
    std::cout << "Matmul Simple took " << timerCPU3.stopTimer() << " milliseconds\r\n";

    long double totalDiff = totalDifference(result3, result1, N, M);
    long double totalDiff2 = totalDifference(result3, result2, N, M);
    std::cout << "Total difference between simple cuda float and first WMMA result: " << totalDiff << std::endl;
    std::cout << "Total difference between simple cuda float and second WMMA result: " << totalDiff2 << std::endl;
    std::cout << "Average difference between regular and first WMMA result: " << totalDiff / (long double)(N * M) << std::endl;
    std::cout << "Average difference between regular and second WMMA result: " << totalDiff2 / (long double)(N * M) << std::endl;
    return 0;
}
