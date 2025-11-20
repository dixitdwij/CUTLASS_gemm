#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <iomanip>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
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

// --- Tuning Template ---
// Allows passing shapes as template arguments
template <
    typename ElementType, 
    typename ThreadblockShape, 
    typename WarpShape, 
    typename InstructionShape, 
    int Stages
>
void run_tuned_gemm(int m, int n, int k) {
    
    using Layout = cutlass::layout::RowMajor;
    using ElementAccumulator = float;

    // Define the Tuned Gemm Operator
    using Gemm = cutlass::gemm::device::Gemm<
        ElementType, Layout,
        ElementType, Layout,
        ElementType, Layout,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp, // Use Tensor Cores
        cutlass::arch::Sm80,            // Target Architecture (Sm90 is backward compat with Sm80 API)
        ThreadblockShape,               // Tunable: Threadblock Size
        WarpShape,                      // Tunable: Warp Size
        InstructionShape,               // Tunable: Instruction Size
        cutlass::epilogue::thread::LinearCombination<
            ElementType, 
            128 / cutlass::sizeof_bits<ElementType>::value,
            ElementAccumulator, 
            ElementAccumulator
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Stages                          // Tunable: Pipeline Stages
    >;

    std::cout << "Running Tuned GEMM (M=" << m << ", N=" << n << ", K=" << k << ")..." << std::endl;
    std::cout << "Config: " 
              << "TB<" << ThreadblockShape::kM << "," << ThreadblockShape::kN << "," << ThreadblockShape::kK << "> "
              << "Warp<" << WarpShape::kM << "," << WarpShape::kN << "," << WarpShape::kK << "> "
              << "Stages=" << Stages 
              << std::endl;

    // Allocations
    ElementType *dev_A, *dev_B, *dev_C;
    size_t size_A = size_t(m) * k * sizeof(ElementType);
    size_t size_B = size_t(k) * n * sizeof(ElementType);
    size_t size_C = size_t(m) * n * sizeof(ElementType);

    CHECK_CUDA(cudaMalloc(&dev_A, size_A));
    CHECK_CUDA(cudaMalloc(&dev_B, size_B));
    CHECK_CUDA(cudaMalloc(&dev_C, size_C));

    // Args
    ElementAccumulator alpha = 1.0f;
    ElementAccumulator beta  = 1.0f;
    typename Gemm::Arguments arguments({m, n, k}, {dev_A, k}, {dev_B, n}, {dev_C, n}, {dev_C, n}, {alpha, beta});

    Gemm gemm_op;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    void *workspace = nullptr;
    if (workspace_size > 0) CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

    CHECK_CUTLASS(gemm_op.initialize(arguments, workspace));

    // Profile
    const int warmup = 10;
    const int iter = 100;

    for(int i=0; i<warmup; ++i) gemm_op(arguments, workspace);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i=0; i<iter; ++i) gemm_op(arguments, workspace);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double avg_sec = (ms / 1000.0) / iter;
    double tflops = (2.0 * double(m) * double(n) * double(k)) / avg_sec / 1.0e12;

    std::cout << "Time: " << std::fixed << std::setprecision(3) << ms/iter << " ms | ";
    std::cout << "Perf: " << tflops << " TFLOPs" << std::endl;

    cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);
    if(workspace) cudaFree(workspace);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K>" << std::endl;
        return -1;
    }
    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    int k = std::atoi(argv[3]);

    // ============================================================================
    // TUNING CONFIGURATION SECTION
    // Change these parameters to tune performance for SM90
    // ============================================================================
    
    // 1. Threadblock Shape: <M, N, K>
    //    Recommended for SM80/90: <128, 256, 32> or <128, 128, 32> or <256, 128, 32>
    using ShapeTB = cutlass::gemm::GemmShape<128, 128, 32>;

    // 2. Warp Shape: <M, N, K>
    //    Standard: <64, 64, 32>
    using ShapeWarp = cutlass::gemm::GemmShape<64, 64, 32>;

    // 3. Instruction Shape: <M, N, K>
    //    Hardcoded for Tensor Cores (bf16) usually <16, 8, 16>
    using ShapeOp = cutlass::gemm::GemmShape<16, 8, 16>;

    // 4. Pipeline Stages
    //    SM80/90 has large register files/shared mem. Try 3, 4, or 5.
    const int Stages = 4;

    // Run
    run_tuned_gemm<cutlass::bfloat16_t, ShapeTB, ShapeWarp, ShapeOp, Stages>(m, n, k);

    return 0;
}