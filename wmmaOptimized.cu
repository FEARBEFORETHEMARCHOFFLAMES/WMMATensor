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
#include "nvtx3/nvToolsExt.h"

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
// Stride in load_matrix_sync must be a multiple of 8: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma-description
// Accessing fragments directly offers no guarantee of element order: https://forums.developer.nvidia.com/t/how-does-the-operation-like-some-fragment-x-index-work-in-wmma-api/287291
// Precision loss due to tensor cores: https://arxiv.org/pdf/1803.04014
// "Pitched" i.e. padded memory on the gpu is initialized to zero in my tests, but i can not find any guarantees/documentation of it!! avoid accessing it for now
// at least 96kb of shared mem(per SM) on consumer grade GPUs https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf

//--------------------------------------------------- Kernels ------------------------------------------------------------------------

// This function is from the TU Dresden Highly Parallel Programming of GPUs Lecture
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

// - only squared dim allowed (width*width)
// - only blockdim = Tilewidth*Tilewidth allowed
// - only dim = x * Tilewidth allowed (multiples of Tilewidth)
// Take the loop structure from the simple GPU kernel and
// split up the k-loop into two loops:
// - loop through the tiles 0,..,nr_tiles_x-1
//   - load block of A into s_a, and block of B into s_b
//   - __syncthreads()
//   - loop to compute the partial dot product
//     - temp += s_a[threadIdx.y][·] * s_b[·][threadIdx.x]
//   - __syncthreads()
// This function is from the TU Dresden Highly Parallel Programming of GPUs Lecture
template<int Tilewidth, typename T>
__global__ void matmul_tiled(T* const c,
    T const* const a,
    T const* const b,
    const int width) {

    // GPU matmul code, using static shared memory
    // (2D array, Tilewidth x Tilewidth)

      // Allocate 2D tiles in shared memory
    __shared__ T s_a[Tilewidth][Tilewidth];
    __shared__ T s_b[Tilewidth][Tilewidth];

    const int nr_tiles_x = width / Tilewidth;

    for (int row = threadIdx.y + blockIdx.y * blockDim.y;
        row < width;
        row += blockDim.y * gridDim.y) {

        for (int col = threadIdx.x + blockIdx.x * blockDim.x;
            col < width;
            col += blockDim.x * gridDim.x) {

            T result = 0;

            // Loop over tiles of input in phases
            for (int p = 0; p < nr_tiles_x; p++) {
                // Collaboratively load tiles into shared memory
                s_a[threadIdx.y][threadIdx.x] = a[row * width + (p * Tilewidth + threadIdx.x)];
                s_b[threadIdx.y][threadIdx.x] = b[(p * Tilewidth + threadIdx.y) * width + col];

                // Wait until all data is loaded before allowing any threads in the block to continue
                __syncthreads();

                // Dot product between row of s_a and column of s_b
                for (int ti = 0; ti < Tilewidth; ti++) {
                    result += s_a[threadIdx.y][ti] * s_b[ti][threadIdx.x];
                }

                // Wait until all calculations are finished before allowing any threads in the block to continue
                __syncthreads();
            }
            // Write result
            c[row * width + col] = result;
        } // col
    } // row
}

using namespace nvcuda::wmma;

// This function expects a 1D grid, 1D block and for there to be exactly one warp for each 16x16 tile in the matrix
template<typename T> __global__ void wmmaFixedAmountOfThreads(T* a, T* b, T* c, const int N, const int K, const int M) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / WARP_SIZE;
    int tileIdX = warpId % (M / TILE_SIZE);
    int tileIdY = warpId / (M / TILE_SIZE);

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

// This function will only take care of the elements that fit neatly into 16x16 tiling.
template<typename T> __global__ void wmmaGridStride(T* a, T* b, T* c, const int N, const int K, const int M) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / WARP_SIZE;
    int tileIdX = warpId % (M / TILE_SIZE);
    int tileIdY = warpId / (M / TILE_SIZE);
    while(tileIdY < (N / TILE_SIZE)) {
        fragment<matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, T, row_major> a_frag;
        fragment<matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, T, row_major> b_frag;
        fragment<accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, T> c_frag;
        fill_fragment(c_frag, 0.0f);
        for (int k = 0; k < K; k += TILE_SIZE) {
            load_matrix_sync(a_frag, &a[tileIdY * TILE_SIZE * K + k], K);
            load_matrix_sync(b_frag, &b[k * M + tileIdX * TILE_SIZE], M);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        store_matrix_sync(&c[tileIdY * TILE_SIZE * M + tileIdX * TILE_SIZE], c_frag, M, mem_row_major);
        
        threadId += blockDim.x * gridDim.x;
        warpId = threadId / WARP_SIZE;
        tileIdX = warpId % (M / TILE_SIZE);
        tileIdY = warpId / (M / TILE_SIZE);
    }
}

// This function will only take care of the elements that fit neatly into 16x16 tiling. Additionally takes "pitch", i.e. padding, of the matrix pointers as arguments.
template<typename T> __global__ void wmmaGridStrideOnTileableArea(T* a, T* b, T* c, const int N, const int K, const int M, size_t pitch_a, size_t pitch_b, size_t pitch_c) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / WARP_SIZE;
    int tileIdX = warpId % (M / TILE_SIZE);
    int tileIdY = warpId / (M / TILE_SIZE);
    while (tileIdY < (N / TILE_SIZE)) {
        fragment<matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, T, row_major> a_frag;
        fragment<matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, T, row_major> b_frag;
        fragment<accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, T> c_frag;
        fill_fragment(c_frag, 0.0f);

        for (int k = 0; k < (K - K % TILE_SIZE); k += TILE_SIZE) {
            load_matrix_sync(a_frag, ((T*)((char*)a + tileIdY * TILE_SIZE * pitch_a + k * sizeof(T))), pitch_a / sizeof(T));
            load_matrix_sync(b_frag, ((T*)((char*)b + k * pitch_b + tileIdX * TILE_SIZE * sizeof(T))), pitch_b / sizeof(T));
            mma_sync(c_frag, a_frag, b_frag, c_frag);

        }
        store_matrix_sync((T*)((char*)c + tileIdY * TILE_SIZE * pitch_c + tileIdX * TILE_SIZE * sizeof(T)), c_frag, pitch_c / sizeof(T), mem_row_major);

        threadId += blockDim.x * gridDim.x;
        warpId = threadId / WARP_SIZE;
        tileIdX = warpId % (M / TILE_SIZE);
        tileIdY = warpId / (M / TILE_SIZE);
    }
}

// Option 1: Go over entire matrix to add results from untileable area.
template<typename T> __global__ void tilingFix(T* a, T* b, T* c, const int N, const int K, const int M, size_t pitch_a, size_t pitch_b, size_t pitch_c) {
    // Phase 1: iterate over elements that have been taken care of during the tiling matrix matrix multiplication.
    // -> these elements lack the values in the "leftover" parts of the matrix.
    int leftoverRows = N - N % TILE_SIZE;
    int leftoverColumns = M - M % TILE_SIZE;
    int leftoverK = K - K % TILE_SIZE;
    for (int id = threadIdx.x + blockIdx.x * blockDim.x; id < leftoverRows * leftoverColumns; id += blockDim.x * gridDim.x) {
        int y = id / leftoverColumns;
        int x = id % leftoverColumns;
        T result = 0;
        for (int k = leftoverK; k < K ; k++) {
            result += *(T*)((char*)a + y * pitch_a + k * sizeof(T)) * *(T*)((char*)b + k * pitch_b + x * sizeof(T));
        }
        *(T*)((char*)c + y * pitch_c + x * sizeof(T)) += result;
    }

    // Phase 2: iteratee over elements that are in the leftover area
    // each element has not been computed at all yet
    // we have exactly one thread per element 
    // 1. determine which element this thread is responsible for
    int globalId = threadIdx.x + blockIdx.x * blockDim.x;
    int x = 0;
    int y = 0;
    if (globalId < (M % TILE_SIZE) * (N % TILE_SIZE)) {
        // we are in the range of IDs that should be assigned to one of the elements in the bottom right of the matrix
        x = (M - M % TILE_SIZE) + globalId % (M % TILE_SIZE);
        y = (N - N % TILE_SIZE) + globalId / (M % TILE_SIZE);
    }
    else if (globalId < (M % TILE_SIZE) * N) {
        // should be assigned to one of the elements in the leftover columns
        x = (M - M % TILE_SIZE) + (globalId - (M % TILE_SIZE) * (N % TILE_SIZE)) % (M % TILE_SIZE);
        y = (globalId - (M % TILE_SIZE) * (N % TILE_SIZE)) / (M % TILE_SIZE);
    }
    else if (globalId < (M % TILE_SIZE) * N + (N % TILE_SIZE) * M) {
        // should be assigned to one of the elemnts in the leftover rows
        x = (globalId - (M % TILE_SIZE) * N) % (M - M % TILE_SIZE);
        y = (N - N % TILE_SIZE) + (globalId - (M % TILE_SIZE) * N) / (M - M % TILE_SIZE);
    }
    else {
        return;
    }
    T cResult = 0;
    for (int k = 0; k < K; k++) {
        T aValue = *(T*)((char*)a + y * pitch_a + k * sizeof(T));
        T bValue = *(T*)((char*)b + k * pitch_b + x * sizeof(T));
        cResult += aValue * bValue;
    }
    *(T*)((char*)c + y * pitch_c + x * sizeof(T)) = cResult;

}

// Option 2: Go over tileable matrix part and add results from untileable area.
template<typename T> __global__ void tilingFixMainMatrix(T* a, T* b, T* c, const int N, const int K, const int M, size_t pitch_a, size_t pitch_b, size_t pitch_c) {
    // Phase 1: iterate over elements that have been taken care of during the tiling matrix matrix multiplication.
    // -> these elements lack the values in the "leftover" parts of the matrix.
    int leftoverRows = N - N % TILE_SIZE;
    int leftoverColumns = M - M % TILE_SIZE;
    int leftoverK = K - K % TILE_SIZE;
    for (int id = threadIdx.x + blockIdx.x * blockDim.x; id < leftoverRows * leftoverColumns; id += blockDim.x * gridDim.x) {
        int y = id / leftoverColumns;
        int x = id % leftoverColumns;
        T result = 0;
        for (int k = leftoverK; k < K; k++) {
            result += *(T*)((char*)a + y * pitch_a + k * sizeof(T)) * *(T*)((char*)b + k * pitch_b + x * sizeof(T));
        }
        *(T*)((char*)c + y * pitch_c + x * sizeof(T)) += result;
    }
}

// Option2: Go over untileable matrix part and compute result.
template<typename T> __global__ void tilingFixLeftovers(T* a, T* b, T* c, const int N, const int K, const int M, size_t pitch_a, size_t pitch_b, size_t pitch_c) {
    // Phase 2: iteratee over elements that are in the leftover area
    // each element has not been computed at all yet
    // we have exactly one thread per element 
    // 1. determine which element this thread is responsible for
    int globalId = threadIdx.x + blockIdx.x * blockDim.x;
    int x = 0;
    int y = 0;
    if (globalId < (M % TILE_SIZE) * (N % TILE_SIZE)) {
        // we are in the range of IDs that should be assigned to one of the elements in the bottom right of the matrix
        x = (M - M % TILE_SIZE) + globalId % (M % TILE_SIZE);
        y = (N - N % TILE_SIZE) + globalId / (M % TILE_SIZE);
    }
    else if (globalId < (M % TILE_SIZE) * N) {
        // should be assigned to one of the elements in the leftover columns
        x = (M - M % TILE_SIZE) + (globalId - (M % TILE_SIZE) * (N % TILE_SIZE)) % (M % TILE_SIZE);
        y = (globalId - (M % TILE_SIZE) * (N % TILE_SIZE)) / (M % TILE_SIZE);
    }
    else if (globalId < (M % TILE_SIZE) * N + (N % TILE_SIZE) * M) {
        // should be assigned to one of the elemnts in the leftover rows
        x = (globalId - (M % TILE_SIZE) * N) % (M - M % TILE_SIZE);
        y = (N - N % TILE_SIZE) + (globalId - (M % TILE_SIZE) * N) / (M - M % TILE_SIZE);
    }
    else {
        return;
    }
    T cResult = 0;
    for (int k = 0; k < K; k++) {
        T aValue = *(T*)((char*)a + y * pitch_a + k * sizeof(T));
        T bValue = *(T*)((char*)b + k * pitch_b + x * sizeof(T));
        cResult += aValue * bValue;
    }
    *(T*)((char*)c + y * pitch_c + x * sizeof(T)) = cResult;
}

//--------------------------------------------------- Setup functions for Kernels ----------------------------------------------------
//                      Each function sets up the memory for the corresponding kernel, calls it and retrieves the result

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

template<typename T> void runMatmulTiled(T* h_a, T* h_b, T* h_c, T* result, const int N, const int K, const int M, const int threadsPerBlock, const int blocks) {
    T* h_d = new T[N * M];
    T* d_a, * d_b, * d_c;
    CHECK_CUDA(cudaMalloc(&d_a, N * K * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_b, K * M * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_c, N * M * sizeof(T)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * K * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * M * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c, N * M * sizeof(T), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock3D(TILE_SIZE, TILE_SIZE);
    if (N != K || K != M) {
        printf("Error in runMatmulTiled. Only square matrices allowed");
    }
    int blocksDim = std::sqrt(threadsPerBlock * blocks / TILE_SIZE);
    if (blocksDim < 1) {
        blocksDim = 1;
    }
    dim3 blocksPerGrid3D(blocksDim, blocksDim);

    matmul_tiled<TILE_SIZE, T> << <blocksPerGrid3D, threadsPerBlock3D >> > (d_c, d_a, d_b, N);

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

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * K * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * M * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c, N * M * sizeof(T), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock3D(threadsPerBlock);
    dim3 blocksPerGrid3D(blocks);

    wmmaFixedAmountOfThreads<T><<<blocksPerGrid3D, threadsPerBlock3D>>>(d_a, d_b, d_c, N, K, M);

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

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * K * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * M * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c, N * M * sizeof(T), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock3D(threadsPerBlock);
    dim3 blocksPerGrid3D(blocks);

    nvtxRangePush("Pretiled");
    wmmaGridStride<T> << <blocksPerGrid3D, threadsPerBlock3D>> > (d_a, d_b, d_c, N, K, M);
    nvtxRangePop();

    CHECK_CUDA(cudaMemcpy(result, d_c, N * M * sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
}

template<typename T> void runWMMATilingFix(T* h_a, T* h_b, T* h_c, T* result, const int N, const int K, const int M, const int threadsPerBlock, const int blocks) {
    T* h_d = new T[N * M];
    T* d_a, * d_b, * d_c;
    size_t pitch_a, pitch_b, pitch_c;
    CHECK_CUDA(cudaMallocPitch((void**)&d_a, &pitch_a, K * sizeof(T), N));
    CHECK_CUDA(cudaMallocPitch((void**)&d_b, &pitch_b, M * sizeof(T), K));
    CHECK_CUDA(cudaMallocPitch((void**)&d_c, &pitch_c, M * sizeof(T), N));

    CHECK_CUDA(cudaMemcpy2D(d_a, pitch_a, h_a, K * sizeof(T), K * sizeof(T), N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy2D(d_b, pitch_b, h_b, M * sizeof(T), M * sizeof(T), K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy2D(d_c, pitch_c, h_c, M * sizeof(T), M * sizeof(T), N, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock3D(threadsPerBlock);
    dim3 blocksPerGrid3D(blocks);
    // Calculate threads for tilingFix. One thread per leftover element.
    // Idea: pass multiple of 32 as total number of threads. Some of these wont be used during phase 2, but can be useful for phase 1. Problem: thread divergence
    int leftoverElements = (M % TILE_SIZE) * N + (N % TILE_SIZE) * M - (M % TILE_SIZE) * (N % TILE_SIZE);
    dim3 threadsPerBLock3DTilingFixLeftovers(64);
    dim3 blocksPerGrid3DTilingFixLeftovers(std::max(1, (int)std::ceil((double)leftoverElements / 64.0f)));

    dim3 threadsPerBlock3DTilingFixMainMatrix(128);
    dim3 blocksPerGrid3DTilingFixMainMatrix(std::max(1, (N - N % TILE_SIZE) * (M - M & TILE_SIZE) / 128));// This is for option 2, i.e. using two kernels for fixing the matrix output

    int priorityLow, priorityHigh;
    CHECK_CUDA(cudaDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh));
    cudaStream_t highPriorityStream, mediumPriorityStream, lowPriorityStream;
    CHECK_CUDA(cudaStreamCreateWithPriority(&highPriorityStream, cudaStreamDefault, priorityHigh));
    CHECK_CUDA(cudaStreamCreateWithPriority(&mediumPriorityStream, cudaStreamDefault, priorityLow + 1));
    CHECK_CUDA(cudaStreamCreateWithPriority(&lowPriorityStream, cudaStreamDefault, priorityLow));

    nvtxRangePush("TileableFix");
    // Kernel launches
    if (leftoverElements == 0) {
        wmmaGridStrideOnTileableArea<T> << <blocksPerGrid3D, threadsPerBlock3D, 0, highPriorityStream >> > (d_a, d_b, d_c, N, K, M, pitch_a, pitch_b, pitch_c);
    }
    else {
        wmmaGridStrideOnTileableArea<T> << <blocksPerGrid3D, threadsPerBlock3D, 0, highPriorityStream >> > (d_a, d_b, d_c, N, K, M, pitch_a, pitch_b, pitch_c);
        tilingFix<T> << <blocksPerGrid3DTilingFixLeftovers, threadsPerBLock3DTilingFixLeftovers, 0, lowPriorityStream >> > (d_a, d_b, d_c, N, K, M, pitch_a, pitch_b, pitch_c);
    }
    nvtxRangePop();

    CHECK_CUDA(cudaMemcpy2D(result, M * sizeof(T), d_c, pitch_c, M * sizeof(T), N, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaStreamSynchronize(highPriorityStream));
    CHECK_CUDA(cudaStreamSynchronize(lowPriorityStream));
    CHECK_CUDA(cudaStreamDestroy(highPriorityStream));
    CHECK_CUDA(cudaStreamDestroy(lowPriorityStream));
}

//--------------------------------------------------- Macro functions ---------------------------------------------------------------
//      Call the setup functions to create test files, get quick data or simply call them to assess them through nsight compute
void createTestFile() {
    const int dev = 0;
    std::cout << getCUDADeviceInformations(dev).str() << "\n\n";
    std::srand(1337);

    int sizes[5] = { 16, 512, 1024, 2048, 4096 };
    int repetitions = 10;
    int N, K, M = 0;
    std::ofstream myfile;
    half divisor = (half)(RAND_MAX + 1u)/* / (half)6.0f*/;
    myfile.open("FloatSimple_vs_HalfWMMA_0_1_StandardDeviation.csv");
    for (int s = 0; s < 5; s++) {
        N = sizes[s];
        M = sizes[s];
        K = sizes[s];
        int tileableM = M + (TILE_SIZE - M % TILE_SIZE);
        int tileableK = K + (TILE_SIZE - K % TILE_SIZE);
        int tileableN = N + (TILE_SIZE - N % TILE_SIZE);
        if (M % TILE_SIZE == 0) { tileableM = M; }
        if (K % TILE_SIZE == 0) { tileableK = K; }
        if (N % TILE_SIZE == 0) { tileableN = N; }
        int totalThreads = WARP_SIZE * (tileableN / TILE_SIZE) * (tileableM / TILE_SIZE);
        int threadsPerBlock = 256;
        int blocks = std::max(1, totalThreads / threadsPerBlock);
        half* h_a = new half[N * K]; // N hoch, K breit
        float* h_a_2 = new float[N * K]; // N hoch, K breit
        half* h_b = new half[K * M]; // K hoch, M breit
        float* h_b_2 = new float[K * M]; // K hoch, M breit
        half* h_c = new half[N * M]; // N hoch, M breit
        float* h_c_2 = new float[N * M]; // N hoch, M breit
        for (int i = 0; i < repetitions; i++) {
            for (size_t i = 0; i < N * K; i++) {
                half randomNumber =/*(half)1.0f + */(half)std::rand() / divisor;
                h_a[i] = randomNumber;
                h_a_2[i] = randomNumber;
            }
            for (size_t i = 0; i < K * M; i++) {
                half randomNumber = /*(half)1.0f + */(half)std::rand() / divisor;
                h_b[i] = randomNumber;
                h_b_2[i] = randomNumber;
            }
            for (size_t i = 0; i < N * M; i++) {
                //It is not possible to load things directly into c in wmma!
                h_c[i] = 0.0f;
                h_c_2[i] = 0.0f;
            }
            std::cout << "Done with generating data" << std::endl;
            half* result1 = new half[N * M];
            runWMMAGridStride<half>(h_a, h_b, h_c, result1, N, K, M, threadsPerBlock, blocks);
            std::cout << "Done with wmma grid stride" << std::endl;
            float* result2 = new float[N * M];
            runMatmulTiled<float>(h_a_2, h_b_2, h_c_2, result2, N, K, M, threadsPerBlock, blocks);
            std::cout << "Done with float tiled" << std::endl;

            long double totalDiff = totalDifference(result2, result1, N, M);
            long double standardDev = standardDeviation(result2, result1, N, M);
            std::cout << "Calculation done for size " << sizes[s] << " and repetition " << i << std::endl;
            std::cout << "Total difference between tiled cuda and WMMA grid stride result: " << totalDiff << std::endl;
            std::cout << "Average difference between tiled cuda and WMMA grid stride result: " << totalDiff / (long double)(N * M) << std::endl;
            std::cout << "Standard deviation between tiled cuda and WMMA grid stride result: " << standardDev << std::endl << std::endl;
            std::cout << "The range of input numbers was half numbers in [0,1] " << std::endl << std::endl;

            myfile << standardDev << ";";
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

void quickDevelopmentData() {
    const int dev = 0;
    std::cout << getCUDADeviceInformations(dev).str() << "\n\n";
    std::srand(1337);
    int M = 1000, K = 500, N = 1000;
    int tileableM = M + (TILE_SIZE - M % TILE_SIZE);
    int tileableK = K + (TILE_SIZE - K % TILE_SIZE);
    int tileableN = N + (TILE_SIZE - N % TILE_SIZE);
    if (M % TILE_SIZE == 0) { tileableM = M; }
    if (K % TILE_SIZE == 0) { tileableK = K; }
    if (N % TILE_SIZE == 0) { tileableN = N; }
    // the number of threads should be 32 * (N/16) * (M/16)
    // adjust blocks such that this holds
    int totalThreads = WARP_SIZE * (tileableN / TILE_SIZE) * (tileableM / TILE_SIZE);
    int threadsPerBlock = 256;
    int blocks = std::max(1, totalThreads / threadsPerBlock);

    half* host_a_half = new half[N * K]; // N hoch, K breit
    half* host_b_half = new half[K * M]; // K hoch, M breit
    half* host_c_half = new half[N * M]; // N hoch, M breit


    half* host_a_half_tileable = new half[tileableN * tileableK]; // N hoch, K breit
    half* host_b_half_tileable = new half[tileableK * tileableM]; // K hoch, M breit
    half* host_c_half_tileable = new half[tileableN * tileableM]; // N hoch, M breit

    float* host_a_float = new float[N * K]; // N hoch, K breit
    float* host_b_float = new float[K * M]; // K hoch, M breit
    float* host_c_float = new float[N * M]; // N hoch, M breit

    half divisor_half = (half)(RAND_MAX + 1u);
    float divisor_double = (float)(RAND_MAX + 1u);

    for (int x = 0; x < K; x++) {
        for (int y = 0; y < N; y++) {
            int randomNumber = std::rand();
            host_a_half[y * K + x] = (half)randomNumber / divisor_half;
            host_a_half_tileable[y * tileableK + x] = host_a_half[y * K + x];
            host_a_float[y * K + x] = (float)randomNumber / divisor_double;
        }
    }

    for (int x = K; x < tileableK; x++) {
        for (int y = 0; y < tileableN; y++) {
            int randomNumber = std::rand();
            host_a_half_tileable[y * tileableK + x] = 0;
        }
    }
    for (int x = 0; x < tileableK; x++) {
        for (int y = N; y < tileableN; y++) {
            int randomNumber = std::rand();
            host_a_half_tileable[y * tileableK + x] = 0;
        }
    }

    std::cout << "Generated a \r\n";

    for (int x = 0; x < M; x++) {
        for (int y = 0; y < K; y++) {
            int randomNumber = std::rand();
            host_b_half[y * M + x] = (half)randomNumber / divisor_half;
            host_b_half_tileable[y * tileableM + x] = host_b_half[y * M + x];
            host_b_float[y * M + x] = (float)randomNumber / divisor_double;
        }
    }

    for (int x = M; x < tileableM; x++) {
        for (int y = 0; y < tileableK; y++) {
            int randomNumber = std::rand();
            host_b_half_tileable[y * tileableM + x] = 0;
        }
    }
    for (int x = 0; x < tileableM; x++) {
        for (int y = K; y < tileableK; y++) {
            int randomNumber = std::rand();
            host_b_half_tileable[y * tileableM + x] = 0;
        }
    }

    for (int i = 0; i < N * M; i++) {
        host_c_half[i] = 0.0f;
        host_c_float[i] = 0.0f;
    }

    for (int i = 0; i < tileableN * tileableM; i++) {
        host_c_half_tileable[i] = 0.0f;
    }

    std::cout << "Generated b \r\n";
    //printMat(host_a_half, N, M, "Input");

    //half* result1 = new half[N * M];
    //TimerCPU timerCPU;
    //timerCPU.startTimer();
    //runWMMAFixedAmountOfThreads<half>(host_a_half, host_b_half, host_c_half, result1, N, K, M, threadsPerBlock, blocks);
    //std::cout << "WMMA fixed amount of threads kernel took " << timerCPU.stopTimer() << " milliseconds\r\n";
    //printMat(result1, N, M, "FixedThreadsResult");

    half* result2 = new half[N * M];
    half* result2WithZeros = new half[tileableN * tileableM];
    TimerCPU timerCPU2;
    timerCPU2.startTimer();
    runWMMAGridStride<half>(host_a_half_tileable, host_b_half_tileable, host_c_half_tileable, result2WithZeros, tileableN, tileableK, tileableM, threadsPerBlock, blocks);
    std::cout << "WMMA kernel with grid stride took " << timerCPU2.stopTimer() << " milliseconds\r\n";
    for (int x = 0; x < M; x++) {
        for (int y = 0; y < N; y++) {
            result2[y * M + x] = result2WithZeros[y * tileableM + x];
        }
    }
    //printMat(result2, N, M, "WWMA Grid Stride Result");

    float* result3 = new float[N * M];
    TimerCPU timerCPU3;
    timerCPU3.startTimer();
    runMatmulSimple<float>(host_a_float, host_b_float, host_c_float, result3, N, K, M, threadsPerBlock, blocks);
    std::cout << "Matmul Tiled took " << timerCPU3.stopTimer() << " milliseconds\r\n";
    //printMat(result3, N, M, "CudaMatmulSimpleResult");

    /*float* result4 = new float[N * M];
    TimerCPU timerCPU4;
    timerCPU4.startTimer();
    runMatmulSimple<float>(host_a_float, host_b_float, host_c_float, result4, N, K, M, threadsPerBlock, blocks);
    std::cout << "Matmul Simple took " << timerCPU4.stopTimer() << " milliseconds\r\n";*/
    //printMat(result4, N, M, "RegularResult");

    half* result5 = new half[N * M];
    TimerCPU timerCPU5;
    timerCPU5.startTimer();
    runWMMATilingFix<half>(host_a_half, host_b_half, host_c_half, result5, N, K, M, threadsPerBlock, blocks);
    std::cout << "Matmul WMMA tiling fix took " << timerCPU5.stopTimer() << " milliseconds\r\n";
    //printMat(result5, N, M, "WMMA Tiling fix Result");

    long double totalDiff = totalDifference(result3, result2, N, M);
    long double totalDiff2 = totalDifference(result3, result5, N, M);
    //long double totalDiff2 = totalDifference(result4, result2, N, M);
    std::cout << "Total difference between tiled cuda float and first WMMA result: " << totalDiff << std::endl;
    std::cout << "Total difference between tiled cuda float and second WMMA result: " << totalDiff2 << std::endl;
    std::cout << "Average difference between tiled cuda and first WMMA result: " << totalDiff / (long double)(N * M) << std::endl;
    std::cout << "Average difference between tiled cuda and second WMMA result: " << totalDiff2 / (long double)(N * M) << std::endl;

}

void nsightComputeRun() {
    const int dev = 0;
    std::cout << getCUDADeviceInformations(dev).str() << "\n\n";
    int M = 2049, K = 2049, N = 2049;
    int tileableM = M + (TILE_SIZE - M % TILE_SIZE);
    int tileableK = K + (TILE_SIZE - K % TILE_SIZE);
    int tileableN = N + (TILE_SIZE - N % TILE_SIZE);
    if (M % TILE_SIZE == 0) { tileableM = M; }
    if (K % TILE_SIZE == 0) { tileableK = K; }
    if (N % TILE_SIZE == 0) { tileableN = N; }
    // the number of threads should be 32 * (N/16) * (M/16)
    // adjust blocks such that this holds
    int totalThreads = WARP_SIZE * (tileableN / TILE_SIZE) * (tileableM / TILE_SIZE);
    int threadsPerBlock = 256;
    int blocks = std::max(1, totalThreads / threadsPerBlock);

    half* host_a_half = new half[N * K]; // N hoch, K breit
    half* host_b_half = new half[K * M]; // K hoch, M breit
    half* host_c_half = new half[N * M]; // N hoch, M breit


    half* host_a_half_tileable = new half[tileableN * tileableK]; // N hoch, K breit
    half* host_b_half_tileable = new half[tileableK * tileableM]; // K hoch, M breit
    half* host_c_half_tileable = new half[tileableN * tileableM]; // N hoch, M breit

    float* host_a_float = new float[N * K]; // N hoch, K breit
    float* host_b_float = new float[K * M]; // K hoch, M breit
    float* host_c_float = new float[N * M]; // N hoch, M breit

    half divisor_half = (half)(RAND_MAX + 1u);
    float divisor_double = (float)(RAND_MAX + 1u);

    for (int x = 0; x < K; x++) {
        for (int y = 0; y < N; y++) {
            int randomNumber = std::rand();
            host_a_half[y * K + x] = (half)randomNumber / divisor_half;
            host_a_half_tileable[y * tileableK + x] = host_a_half[y * K + x];
            host_a_float[y * K + x] = (float)randomNumber / divisor_double;
        }
    }

    for (int x = K; x < tileableK; x++) {
        for (int y = 0; y < tileableN; y++) {
            int randomNumber = std::rand();
            host_a_half_tileable[y * tileableK + x] = 0;
        }
    }
    for (int x = 0; x < tileableK; x++) {
        for (int y = N; y < tileableN; y++) {
            int randomNumber = std::rand();
            host_a_half_tileable[y * tileableK + x] = 0;
        }
    }

    for (int x = 0; x < M; x++) {
        for (int y = 0; y < K; y++) {
            int randomNumber = std::rand();
            host_b_half[y * M + x] = (half)randomNumber / divisor_half;
            host_b_half_tileable[y * tileableM + x] = host_b_half[y * M + x];
            host_b_float[y * M + x] = (float)randomNumber / divisor_double;
        }
    }

    for (int x = M; x < tileableM; x++) {
        for (int y = 0; y < tileableK; y++) {
            int randomNumber = std::rand();
            host_b_half_tileable[y * tileableM + x] = 0;
        }
    }
    for (int x = 0; x < tileableM; x++) {
        for (int y = K; y < tileableK; y++) {
            int randomNumber = std::rand();
            host_b_half_tileable[y * tileableM + x] = 0;
        }
    }

    for (int i = 0; i < N * M; i++) {
        host_c_half[i] = 0.0f;
        host_c_float[i] = 0.0f;
    }

    for (int i = 0; i < tileableN * tileableM; i++) {
        host_c_half_tileable[i] = 0.0f;
    }


    half* result1 = new half[N * M];
    runWMMATilingFix<half>(host_a_half, host_b_half, host_c_half, result1, N, K, M, threadsPerBlock, blocks);

    half* result2WithZeros = new half[tileableN * tileableM];
    runWMMAGridStride<half>(host_a_half_tileable, host_b_half_tileable, host_c_half_tileable, result2WithZeros, tileableN, tileableK, tileableM, threadsPerBlock, blocks);

    float* result3 = new float[N * M];
    runMatmulTiled<float>(host_a_float, host_b_float, host_c_float, result3, N, K, M, threadsPerBlock, blocks);
}

int main()
{
    nsightComputeRun();
    return 0;
}
