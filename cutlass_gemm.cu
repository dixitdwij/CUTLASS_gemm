#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <iomanip>

// CUTLASS Includes
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h" // Required for bfloat16_t
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

// Templated function covers f32, f16, and bf16
template <typename ElementType>
void run_gemm_profile(int m, int n, int k) {
    using Layout = cutlass::layout::RowMajor;
    
    // Configure GEMM (Input types, Output types, Accumulator type)
    // Note: For bf16/f16, accumulation is typically done in f32 for accuracy
    using ElementAccumulator = float; 

    using Gemm = cutlass::gemm::device::Gemm<
        ElementType, Layout,        // A
        ElementType, Layout,        // B
        ElementType, Layout,        // C (Output)
        ElementAccumulator          // Accumulation type (usually f32)
    >;

    std::cout << "Initializing GEMM (M=" << m << ", N=" << n << ", K=" << k << ")..." << std::endl;

    // --- Memory Allocation ---
    // Note: allocating bytes based on sizeof(ElementType)
    ElementType *dev_A, *dev_B, *dev_C;
    size_t size_A = size_t(m) * k * sizeof(ElementType);
    size_t size_B = size_t(k) * n * sizeof(ElementType);
    size_t size_C = size_t(m) * n * sizeof(ElementType);

    CHECK_CUDA(cudaMalloc(&dev_A, size_A));
    CHECK_CUDA(cudaMalloc(&dev_B, size_B));
    CHECK_CUDA(cudaMalloc(&dev_C, size_C));

    // Initialize with dummy data (conceptually) - skipped for pure profiling speed
    // In a real app, you would cudaMemcpy or run a kernel to init these.

    // --- Arguments ---
    // C += A * B  => alpha=1, beta=1
    // Note: Scalars (alpha/beta) are passed as Accumulator Type (float)
    ElementAccumulator alpha = 1.0f;
    ElementAccumulator beta  = 1.0f;

    int lda = k;
    int ldb = n;
    int ldc = n;

    typename Gemm::Arguments arguments(
        {m, n, k},      // Problem size
        {dev_A, lda},   // Tensor A
        {dev_B, ldb},   // Tensor B
        {dev_C, ldc},   // Tensor C
        {dev_C, ldc},   // Tensor D (Destination)
        {alpha, beta}   // Scalars
    );

    Gemm gemm_op;
    
    // Workspace
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    void *workspace = nullptr;
    if (workspace_size > 0) CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

    CHECK_CUTLASS(gemm_op.initialize(arguments, workspace));

    // --- Profiling ---
    const int warmup = 10;
    const int iter = 100;

    for(int i = 0; i < warmup; ++i) {
        CHECK_CUTLASS(gemm_op(arguments, workspace));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i = 0; i < iter; ++i) {
        CHECK_CUTLASS(gemm_op(arguments, workspace));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double seconds = (ms / 1000.0) / iter;

    // TFLOPs = 2 * M * N * K / Time
    double ops = 2.0 * double(m) * double(n) * double(k);
    double tflops = (ops / seconds) / 1.0e12;

    std::cout << "Type size: " << sizeof(ElementType) << " bytes" << std::endl;
    std::cout << "Avg Time:  " << std::fixed << std::setprecision(3) << ms/iter << " ms" << std::endl;
    std::cout << "Perf:      " << std::setprecision(2) << tflops << " TFLOPs" << std::endl;

    cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);
    if(workspace) cudaFree(workspace);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K> <type: f32|f16|bf16>" << std::endl;
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
    } else if (type == "bf16") {
        run_gemm_profile<cutlass::bfloat16_t>(m, n, k);
    } else {
        std::cerr << "Unsupported type. Use fp32, fp16, or bf16." << std::endl;
        return -1;
    }

    return 0;
}