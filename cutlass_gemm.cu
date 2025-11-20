#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <iomanip>

// CUTLASS Includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUTLASS(status) { \
    if ((status) != cutlass::Status::kSuccess) { \
        std::cerr << "CUTLASS Error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

template <typename ElementType>
void run_gemm_profile(int m, int n, int k) {
    //  Configuration 
    // Standard RowMajor Layout (M*K, K*N, M*N)
    using Layout = cutlass::layout::RowMajor;
    
    // CUTLASS Device Gemm Operator
    using Gemm = cutlass::gemm::device::Gemm<
        ElementType, Layout,
        ElementType, Layout,
        ElementType, Layout
    >;

    std::cout << "Initializing GEMM (M=" << m << ", N=" << n << ", K=" << k << ")..." << std::endl;

    //  Memory Allocation 
    // Using arbitrary data for profiling purposes
    ElementType *dev_A, *dev_B, *dev_C;
    CHECK_CUDA(cudaMalloc(&dev_A, size_t(m) * k * sizeof(ElementType)));
    CHECK_CUDA(cudaMalloc(&dev_B, size_t(k) * n * sizeof(ElementType)));
    CHECK_CUDA(cudaMalloc(&dev_C, size_t(m) * n * sizeof(ElementType)));

    //  Setup Arguments 
    // alpha = 1, beta = 1 implements C += A * B
    ElementType alpha = static_cast<ElementType>(1.0f);
    ElementType beta  = static_cast<ElementType>(1.0f);

    typename Gemm::Arguments arguments(
        {m, n, k},      // Problem size
        {dev_A, k},     // Tensor A (ptr, lda)
        {dev_B, n},     // Tensor B (ptr, ldb)
        {dev_C, n},     // Tensor C (ptr, ldc)
        {dev_C, n},     // Tensor D (ptr, ldd) - same as C
        {alpha, beta}   // Scalars
    );

    Gemm gemm_op;
    
    // Workspace allocation (needed for some split-k kernels)
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    void *workspace = nullptr;
    if (workspace_size > 0) CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

    CHECK_CUTLASS(gemm_op.initialize(arguments, workspace));

    //  Profiling 
    // Constants for profiling
    const int warmup_iterations = 10;
    const int profile_iterations = 100;

    // Warmup
    for(int i = 0; i < warmup_iterations; ++i) {
        CHECK_CUTLASS(gemm_op(arguments, workspace));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Create Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record Start
    cudaEventRecord(start);

    // Run Loop
    for(int i = 0; i < profile_iterations; ++i) {
        CHECK_CUTLASS(gemm_op(arguments, workspace));
    }

    // Record Stop
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate Time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    double avg_time_sec = (milliseconds / 1000.0) / profile_iterations;

    //  TFLOPs Calculation 
    // FLOPs = 2 * M * N * K (multiply + add)
    double total_ops = 2.0 * double(m) * double(n) * double(k);
    double tflops = (total_ops / avg_time_sec) / 1.0e12;

    std::cout << "--" << std::endl;
    std::cout << "Performance Results:" << std::endl;
    std::cout << "Avg Time: " << std::fixed << std::setprecision(4) << avg_time_sec * 1000.0 << " ms" << std::endl;
    std::cout << "Throughput: " << std::setprecision(2) << tflops << " TFLOPs" << std::endl;
    std::cout << "--" << std::endl;

    // Cleanup
    cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);
    if(workspace) cudaFree(workspace);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K> <type: f32|f16>" << std::endl;
        return -1;
    }

    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    int k = std::atoi(argv[3]);
    std::string type = argv[4];

    if (type == "fp32") {
        run_gemm_profile<float>(m, n, k);
    } else if (type == "fp16") {
        run_gemm_profile<cutlass::half_t>(m, n, k);
    } else {
        std::cerr << "Unsupported type. Use fp32 or fp16." << std::endl;
        return -1;
    }

    return 0;
}