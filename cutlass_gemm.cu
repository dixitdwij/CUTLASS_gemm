#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

// CUTLASS Includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

// Error handling helper
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

// Templated function to handle different data types
template <typename ElementType>
void run_gemm(int m, int n, int k) {
    // 1. Define the GEMM Setup
    // RowMajor Layout is standard for C++ (M x K, K x N, M x N)
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::RowMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    // Define the Gemm Operator
    // <ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutOutput>
    using Gemm = cutlass::gemm::device::Gemm<
        ElementType, LayoutInputA,
        ElementType, LayoutInputB,
        ElementType, LayoutOutput
    >;

    std::cout << "Running GEMM (M=" << m << ", N=" << n << ", K=" << k << ")..." << std::endl;

    // 2. Allocate Host Memory
    std::vector<ElementType> host_A(m * k);
    std::vector<ElementType> host_B(k * n);
    std::vector<ElementType> host_C(m * n);

    // Initialize with arbitrary data (e.g., 1.0)
    for(auto &v : host_A) v = static_cast<ElementType>(1.0f);
    for(auto &v : host_B) v = static_cast<ElementType>(1.0f);
    for(auto &v : host_C) v = static_cast<ElementType>(0.0f); // Start C at 0

    // 3. Allocate Device Memory
    ElementType *dev_A, *dev_B, *dev_C;
    CHECK_CUDA(cudaMalloc(&dev_A, m * k * sizeof(ElementType)));
    CHECK_CUDA(cudaMalloc(&dev_B, k * n * sizeof(ElementType)));
    CHECK_CUDA(cudaMalloc(&dev_C, m * n * sizeof(ElementType)));

    // 4. Copy Host to Device
    CHECK_CUDA(cudaMemcpy(dev_A, host_A.data(), m * k * sizeof(ElementType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_B, host_B.data(), k * n * sizeof(ElementType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_C, host_C.data(), m * n * sizeof(ElementType), cudaMemcpyHostToDevice));

    // 5. Setup Arguments
    // Alpha = 1.0, Beta = 1.0 enforces C = (A * B) + C
    ElementType alpha = static_cast<ElementType>(1.0f);
    ElementType beta  = static_cast<ElementType>(1.0f);

    // Leading dimensions (stride between rows for RowMajor)
    int lda = k;
    int ldb = n;
    int ldc = n;

    typename Gemm::Arguments arguments(
        {m, n, k},  // Problem size
        {dev_A, lda}, // Tensor A
        {dev_B, ldb}, // Tensor B
        {dev_C, ldc}, // Tensor C
        {dev_C, ldc}, // Tensor D (Destination, usually same as C)
        {alpha, beta} // Scalars
    );

    // 6. Launch Kernel
    Gemm gemm_op;
    
    // Query workspace size needs
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    }

    // Initialize and Run
    CHECK_CUTLASS(gemm_op.initialize(arguments, workspace));
    CHECK_CUTLASS(gemm_op(arguments, workspace));

    // 7. Sync and cleanup
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // (Optional) Copy back to check results
    // CHECK_CUDA(cudaMemcpy(host_C.data(), dev_C, m * n * sizeof(ElementType), cudaMemcpyDeviceToHost));

    std::cout << "GEMM Completed Successfully." << std::endl;

    cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);
    if(workspace) cudaFree(workspace);
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

    if (type == "f32") {
        run_gemm<float>(m, n, k);
    } else if (type == "f16") {
        run_gemm<cutlass::half_t>(m, n, k);
    } else {
        std::cerr << "Unsupported type. Use f32 or f16." << std::endl;
        return -1;
    }

    return 0;
}