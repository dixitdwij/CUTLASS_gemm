#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <iomanip>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm.h"

// Configuration for benchmarking
#define WARMUP_RUNS 10
#define MEASURED_RUNS 100

// 1. The Core GEMM Kernel (Templated)
template <
    typename ShapeTB, 
    typename ShapeWarp, 
    int Stages
>
void run_gemm_core(int m, int n, int k) {
    
    // Fixed Types for SM90/Blackwell BF16
    using ElementType = cutlass::bfloat16_t;
    using Layout = cutlass::layout::RowMajor;
    using ElementAccumulator = float;
    using ShapeOp = cutlass::gemm::GemmShape<16, 8, 16>; // Fixed for Tensor Cores

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
        Stages
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
              << " Stages: " << Stages << "\n"
              << "  [Result] Time: " << std::fixed << std::setprecision(3) << ms/100.0 << " ms | "
              << "TFLOPs: " << tflops << std::endl;
    std::cout << "==========================================\n";

    cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C); 
    if(ws) cudaFree(ws);
}

// 2. Dispatch Logic (The Magic)

// Macro to reduce boilerplate code
// If user inputs match the macro arguments, instantiate that specific template
#define DISPATCH_GEMM(TB_M, TB_N, TB_K, W_M, W_N, W_K, STAGES) \
    if (tb_m == TB_M && tb_n == TB_N && tb_k == TB_K && \
        w_m == W_M && w_n == W_N && w_k == W_K && stages == STAGES) { \
        run_gemm_core< \
            cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>, \
            cutlass::gemm::GemmShape<W_M, W_N, W_K>, \
            STAGES \
        >(m, n, k); \
        return; \
    }

void dispatch(int m, int n, int k, 
              int tb_m, int tb_n, int tb_k, 
              int w_m, int w_n, int w_k, 
              int stages) {

    // List of Valid Configurations to Compile
    // Manually list every combination to be supported.
    
    // Shape Set 1: 128x256x32
    DISPATCH_GEMM(128, 256, 32,  64, 64, 32,  2);
    DISPATCH_GEMM(128, 256, 32,  64, 64, 32,  3);
    DISPATCH_GEMM(128, 256, 32,  64, 64, 32,  4);
    DISPATCH_GEMM(128, 256, 32,  64, 64, 32,  5);
    DISPATCH_GEMM(128, 256, 32,  64, 64, 32,  6);
    DISPATCH_GEMM(128, 256, 32,  64, 64, 32,  7);
    DISPATCH_GEMM(128, 256, 32,  64, 64, 32,  8);    

    // Shape Set 2: 256x128x32
    DISPATCH_GEMM(256, 128, 32,  64, 64, 32,  2);
    DISPATCH_GEMM(256, 128, 32,  64, 64, 32,  3);
    DISPATCH_GEMM(256, 128, 32,  64, 64, 32,  4);
    DISPATCH_GEMM(256, 128, 32,  64, 64, 32,  5);
    DISPATCH_GEMM(256, 128, 32,  64, 64, 32,  6);
    DISPATCH_GEMM(256, 128, 32,  64, 64, 32,  7);
    DISPATCH_GEMM(256, 128, 32,  64, 64, 32,  8);    

    // Shape Set 3: 128x128x32 (Smaller tile)
    DISPATCH_GEMM(128, 128, 32,  64, 64, 32,  2);
    DISPATCH_GEMM(128, 128, 32,  64, 64, 32,  3);
    DISPATCH_GEMM(128, 128, 32,  64, 64, 32,  4);
    DISPATCH_GEMM(128, 128, 32,  64, 64, 32,  5);
    DISPATCH_GEMM(128, 128, 32,  64, 64, 32,  6);
    DISPATCH_GEMM(128, 128, 32,  64, 64, 32,  7);
    DISPATCH_GEMM(128, 128, 32,  64, 64, 32,  8); 
    
    DISPATCH_GEMM(128, 64, 32,  32, 32, 32,  4);    

    // If we get here, the user asked for a config we didn't pre-compile
    std::cerr << "Error: Configuration not supported (was not pre-compiled)." << std::endl;
    std::cerr << "Requested TB: <" << tb_m << "," << tb_n << "," << tb_k << ">" << std::endl;
    std::cerr << "Supported TBs: <128,256,32>, <256,128,32>, <128,128,32>" << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 11) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K> <TB_M> <TB_N> <TB_K> <W_M> <W_N> <W_K> <Stages>" << std::endl;
        std::cerr << "Example: " << argv[0] << " 4096 4096 4096 128 256 32 64 64 32 4" << std::endl;
        return -1;
    }

    // User Input
    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    int k = std::atoi(argv[3]);

    int tb_m = std::atoi(argv[4]);
    int tb_n = std::atoi(argv[5]);
    int tb_k = std::atoi(argv[6]);

    int w_m = std::atoi(argv[7]);
    int w_n = std::atoi(argv[8]);
    int w_k = std::atoi(argv[9]);

    int stages = std::atoi(argv[10]);

    // Dispatcher
    dispatch(m, n, k, tb_m, tb_n, tb_k, w_m, w_n, w_k, stages);

    return 0;
}