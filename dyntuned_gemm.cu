#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <iomanip>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm.h"

// 0. Macros for Static Parameters
#define REQUIRED_MACROS \
    X(TB_M) \
    X(TB_N) \
    X(TB_K) \
    X(W_M)  \
    X(W_N)  \
    X(W_K)  \
    X(STAGES)

#define X(name) [[maybe_unused]] static const int _check_defined_##name = name;

REQUIRED_MACROS

#undef X

// Configuration for benchmarking
#define WARMUP_RUNS 10
#define MEASURED_RUNS 100

// 1. The Core GEMM Kernel (Templated)
void run_gemm_core(int m, int n, int k) {
    
    // Fixed Types for SM90/Blackwell BF16
    using ElementType = cutlass::bfloat16_t;
    using Layout = cutlass::layout::RowMajor;
    using ElementAccumulator = float;
    using ShapeOp = cutlass::gemm::GemmShape<16, 8, 16>; // Fixed for Tensor Cores
    using ShapeTB = cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>;
    using ShapeWarp = cutlass::gemm::GemmShape<W_M, W_N, W_K>;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementType, Layout,
        ElementType, Layout,
        ElementType, Layout,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ShapeTB,
        ShapeWarp,
        ShapeOp,
        cutlass::epilogue::thread::LinearCombination<
            ElementType, 128 / cutlass::sizeof_bits<ElementType>::value,
            ElementAccumulator, ElementAccumulator
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        STAGES
    >;

    // Setup & Run
    ElementType *dev_A, *dev_B, *dev_C;
    size_t size_A = size_t(m) * k * sizeof(ElementType);
    size_t size_B = size_t(k) * n * sizeof(ElementType);
    size_t size_C = size_t(m) * n * sizeof(ElementType);

    cudaMalloc(&dev_A, size_A);
    cudaMalloc(&dev_B, size_B);
    cudaMalloc(&dev_C, size_C);

    typename Gemm::Arguments arguments(
        {m, n, k}, {dev_A, k}, {dev_B, n}, {dev_C, n}, {dev_C, n}, {1.0f, 1.0f}
    );

    Gemm gemm_op;
    size_t ws_size = Gemm::get_workspace_size(arguments);
    void *ws = nullptr;
    if (ws_size > 0) cudaMalloc(&ws, ws_size);

    if(gemm_op.initialize(arguments, ws) != cutlass::Status::kSuccess) {
        std::cerr << "Failed to initialize CUTLASS kernel." << std::endl;
        exit(1);
    }

    // Profile
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Warmup
    for(int i=0; i<WARMUP_RUNS; ++i) gemm_op(arguments, ws);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i=0; i<MEASURED_RUNS; ++i) gemm_op(arguments, ws);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double tflops = (2.0 * m * n * k) / ((ms/100.0) / 1000.0) / 1e12;

    std::cout << "==========================================\n";
    std::cout << "DataType    : " << "BF16" << "\n";
    std::cout << "Runs        : " << WARMUP_RUNS << " warmup + " << MEASURED_RUNS << " measured\n";
    std::cout << "Matrix Size : " << m << " x " << n << " x " << k << "\n";
    std::cout << "  [Config] TB: <" << ShapeTB::kM << "," << ShapeTB::kN << "," << ShapeTB::kK << ">"
              << " Warp: <" << ShapeWarp::kM << "," << ShapeWarp::kN << "," << ShapeWarp::kK << ">"
              << " Stages: " << STAGES << "\n"
              << "  [Result] Time: " << std::fixed << std::setprecision(3) << ms/100.0 << " ms | "
              << "TFLOPs: " << tflops << std::endl;
    std::cout << "==========================================\n";

    cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C); 
    if(ws) cudaFree(ws);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K>" << std::endl;
        std::cerr << "Expecting <TB_M> <TB_N> <TB_K> <W_M> <W_N> <W_K> <Stages> as compile-time arguments" << std::endl;
        return -1;
    }

    // User Input
    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    int k = std::atoi(argv[3]);

    // Dispatcher
    run_gemm_core(m, n, k);

    return 0;
}