#ifndef CUDA_HELPER_HPP_
#define CUDA_HELPER_HPP_

#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include "mma.h";
#include <iomanip>

#ifndef CUDA_DISABLE_ERROR_CHECKING
#define CHECK_CUDA(ans) check_cuda((ans), "", #ans, __FILE__, __LINE__)
#define CHECK_LAST(msg) check_cuda(cudaGetLastError(), msg, "CHECK_LAST", __FILE__, __LINE__)
#else
#define CHECK_CUDA(ans) {}
#define CHECK_LAST(msg) {}
#endif

inline
void throw_error(int code,
    const char* error_string,
    const char* msg,
    const char* func,
    const char* file,
    int line) {
    throw std::runtime_error("CUDA error "
        + std::string(msg)
        + " " + std::string(error_string)
        + " [" + std::to_string(code) + "]"
        + " " + std::string(file)
        + ":" + std::to_string(line)
        + " " + std::string(func)
    );
}

inline
void check_cuda(cudaError_t code, const char* msg, const char* func, const char* file, int line) {
    if (code != cudaSuccess) {
        throw_error(static_cast<int>(code),
            cudaGetErrorString(code), msg, func, file, line);
    }
}

inline
std::uint32_t get_num_cores(cudaDeviceProp devProp)
{
    std::uint32_t cores = 0;
    std::uint32_t numSMs = devProp.multiProcessorCount;
    switch (devProp.major) {
    case 2: // Fermi
        if (devProp.minor == 1) cores = 48;
        else cores = 32;
        break;
    case 3: // Kepler
        cores = 192;
        break;
    case 5: // Maxwell
        cores = 128;
        break;
    case 6: // Pascal
        if (devProp.minor == 1) cores = 128;
        else if (devProp.minor == 0) cores = 64;
        else std::cerr << "Unknown device type\n";
        break;
    case 7: // Volta
        if (devProp.minor < 5) cores = 64;
        else std::cerr << "Unknown device type\n";
        break;
    default:
        std::cerr << "Unknown device type\n";
        break;
    }
    return numSMs * cores;
}

inline
std::stringstream getCUDADeviceInformations(int dev, const char* sep = "\n") {
    std::stringstream info;
    cudaDeviceProp prop;
    int runtimeVersion = 0;
    size_t f = 0, t = 0;
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    CHECK_CUDA(cudaRuntimeGetVersion(&runtimeVersion));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    CHECK_CUDA(cudaMemGetInfo(&f, &t));
    info << '"' << prop.name << '"'
        << sep << "CC, " << prop.major << '.' << prop.minor
        << sep << "Multiprocessors, " << prop.multiProcessorCount
        << sep << "Memory [MiB], " << t / 1048576
        << sep << "MemoryFree [MiB], " << f / 1048576
        << sep << "MemClock [MHz], " << prop.memoryClockRate / 1000
        << sep << "GPUClock [MHz], " << prop.clockRate / 1000
        << sep << "NumCores, " << get_num_cores(prop)
        << sep << "GFLOPs (FMAD), " << 2 * (prop.clockRate / 1000 * get_num_cores(prop)) / 1000
        << sep << "CUDA Runtime, " << runtimeVersion
        << sep << "Time, "
        << std::put_time(std::localtime(&now), "%F %T")
        ;
    return info;
}

inline std::stringstream listCudaDevices() {
    std::stringstream info;
    int nrdev = 0;
    CHECK_CUDA(cudaGetDeviceCount(&nrdev));
    if (nrdev == 0)
        throw std::runtime_error("No CUDA capable device found");
    for (int i = 0; i < nrdev; ++i)
        info << "ID," << i << "," << getCUDADeviceInformations(i, ",").str() << std::endl;
    return info;
}

template<typename T> inline void printMat(T* A, int height, int width, std::string name) {
    std::cout << name << ": " << "\r\n";
    std::cout << "[ ";
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::cout << __half2float(A[y * width + x ]) << " ";
        }
        if (y == height - 1) {
            std::cout << " ]" << "\r\n";
        }
        else {
            std::cout << "\r\n" << "  ";
        }
    }
}

template<typename T> inline bool areEqual(T* A, T* B, int height, int width) {
    for (int i = 0; i < height * width; i++) {
        if (A[i] != B[i]) {
            std::cout << "difference at " << i << "\r\n";
            std::cout << "A:" << __half2float(A[i]) <<" B:" << __half2float(B[i]) << "\r\n";
            return false;
        }
    }
    return true;
}

template<typename T> inline long double totalDifference(T* A, T* B, int height, int width) {
    long double difference = 0.0f;
    for (int i = 0; i < height * width; i++) {
        float diff = static_cast<float>(A[i]) - static_cast<float>(B[i]);
        if (diff < 0.0f) {
            difference -= static_cast<long double>(diff);
        }
        else {
            difference += static_cast<long double>(diff);
        }
    }
    return difference;
}

inline long double totalDifference(double* A, half* B, int height, int width) {
    long double difference = 0.0f;
    for (int i = 0; i < height * width; i++) {
        double diff = A[i] - static_cast<double>(B[i]);
        if (diff < 0.0f) {
            difference -= static_cast<long double>(diff);
        }
        else {
            difference += static_cast<long double>(diff);
        }
    }
    return difference;
}

inline long double totalDifference(float* A, half* B, int height, int width) {
    long double difference = 0.0f;
    for (int i = 0; i < height * width; i++) {
        float diff = A[i] - static_cast<float>(B[i]);
        if (diff < 0.0f) {
            difference -= static_cast<long double>(diff);
        }
        else {
            difference += static_cast<long double>(diff);
        }
    }
    return difference;
}

inline long double standardDeviation(double* A, half* B, int height, int width) {
    double* differences = new double[height * width];
    long double meanWithSign = 0.0f;
    for (int i = 0; i < height * width; i++) {
        double signedDifference = A[i] - static_cast<double>(B[i]);
        meanWithSign += signedDifference;
        differences[i] = signedDifference;
    }
    meanWithSign = meanWithSign / (height * width);
    
    long double variance = 0.0f;
    for (int i = 0; i < height * width; i++) {
        variance += std::pow(differences[i] - meanWithSign, 2);
    }
    variance = variance / (height * width);

    return std::sqrt(variance);
}

inline long double standardDeviation(float* A, half* B, int height, int width) {
    double* differences = new double[height * width];
    long double meanWithSign = 0.0f;
    for (int i = 0; i < height * width; i++) {
        double signedDifference = A[i] - static_cast<float>(B[i]);
        meanWithSign += signedDifference;
        differences[i] = signedDifference;
    }
    meanWithSign = meanWithSign / (height * width);

    long double variance = 0.0f;
    for (int i = 0; i < height * width; i++) {
        variance += std::pow(differences[i] - meanWithSign, 2);
    }
    variance = variance / (height * width);

    return std::sqrt(variance);
}

/** CPU Wall timer
 */
struct TimerCPU {
    using clock = std::chrono::high_resolution_clock;

    clock::time_point start;
    double time = 0.0;

    void startTimer() {
        start = clock::now();
    }

    double stopTimer() {
        auto diff = clock::now() - start;
        return (time = std::chrono::duration<double, std::milli>(diff).count());
    }
};

#endif
