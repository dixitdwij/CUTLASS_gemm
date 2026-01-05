#include <iostream>
#include <vector>
#include <cstdlib>

// CUTLASS Includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

void parse_cmd_line(int argc, char** argv, int& m, int& n, int& k) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <m> <n> <k>" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    m = std::atoi(argv[1]);
    n = std::atoi(argv[2]);
    k = std::atoi(argv[3]);
}

const int WARMUP_RUNS = 10;
const int MEASUREMENT_RUNS = 50;

int main(int argc, char** argv) {
    int M, N, K;
    parse_cmd_line(argc, argv, M, N, K);

    std::cout << "Benchmarking GEMM on RTX 5060 Ti (Auto-Tuned)" << std::endl;
    std::cout << "Dimensions: M=" << M << " N=" << N << " K=" << K << std::endl;

    // 1. Data Types
    using ElementInputA = cutlass::half_t;
    using ElementInputB = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = float;
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::RowMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    // 2. Define the Kernel
    // STRATEGY: We switch to 'cutlass::arch::Sm75' (Turing).
    // This allows CUTLASS to auto-select default shapes (usually 128x64 or 64x64) 
    // that are guaranteed to fit in the Shared Memory of consumer cards (5060 Ti).
    // Using 'Sm80' defaults here would crash because they assume A100-class memory (100KB+).
    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp, // Uses Tensor Cores
        cutlass::arch::Sm75             // Safe defaults for Consumer GPUs
    >;

    // 3. Allocate Data
    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a({M, K});
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b({K, N});
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c({M, N});

    // 4. Initialize Data
    cutlass::reference::host::TensorFillRandomUniform(tensor_a.host_view(), 1, ElementInputA(2), ElementInputA(-2), 0);
    cutlass::reference::host::TensorFillRandomUniform(tensor_b.host_view(), 1, ElementInputB(2), ElementInputB(-2), 0);
    cutlass::reference::host::TensorFillRandomUniform(tensor_c.host_view(), 1, ElementOutput(2), ElementOutput(-2), 0);

    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();

    // 5. Arguments
    float alpha = 1.0f;
    float beta = 1.0f; 

    typename Gemm::Arguments arguments{
        {M, N, K},
        tensor_a.device_ref(),
        tensor_b.device_ref(),
        tensor_c.device_ref(),
        tensor_c.device_ref(),
        {alpha, beta}
    };

    Gemm gemm_op;

    // 6. Workspace Handling (Always good practice)
    size_t workspace_size = gemm_op.get_workspace_size(arguments);
    void* device_workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t err = cudaMalloc(&device_workspace, workspace_size);
        if(err != cudaSuccess) {
            std::cerr << "Workspace Malloc failed: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // 7. Initialize
    cutlass::Status status = gemm_op.initialize(arguments, device_workspace);

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS Initialization Failed: " << cutlassGetStatusString(status) << std::endl;
        return -1;
    }

    // 8. Warmup
    for(int i = 0; i < WARMUP_RUNS; ++i) {
        status = gemm_op();
        if(status != cutlass::Status::kSuccess) {
            std::cerr << "Warmup failed: " << cutlassGetStatusString(status) << std::endl;
            return -1;
        }
    }
    cudaDeviceSynchronize();

    // 9. Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i = 0; i < MEASUREMENT_RUNS; ++i) {
        gemm_op();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_ms = milliseconds / MEASUREMENT_RUNS;
    
    double flops = 2.0 * double(M) * double(N) * double(K);
    double tflops = flops / (avg_ms / 1000.0) / 1e12;

    std::cout << "Throughput: " << tflops << " TFLOPs" << std::endl;

    if (device_workspace) cudaFree(device_workspace);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}