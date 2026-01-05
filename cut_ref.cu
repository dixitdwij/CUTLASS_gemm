#include <iostream>
#include <vector>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/host_tensor.h"

// 1. Define the configuration (Using defaults for SM80/Ampere)
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

// GemmUniversal detects the architecture and picks "best-practice" defaults
using Gemm = cutlass::gemm::device::GemmUniversal<
    ElementA, cutlass::layout::RowMajor,
    ElementB, cutlass::layout::ColumnMajor,
    ElementC, cutlass::layout::RowMajor,
    ElementAccumulator
>;

int main() {
    int M = 4096, N = 4096, K = 4096;
    int warmup_runs = 5;
    int test_runs = 10;

    // Allocate tensors
    cutlass::HostTensor<ElementA, cutlass::layout::RowMajor> tensor_a({M, K});
    cutlass::HostTensor<ElementB, cutlass::layout::ColumnMajor> tensor_b({K, N});
    cutlass::HostTensor<ElementC, cutlass::layout::RowMajor> tensor_c({M, N});

    // Initialize arguments
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K}, 1, {1.0f, 0.0f},
        tensor_a.device_data(), tensor_b.device_data(),
        tensor_c.device_data(), tensor_c.device_data(),
        M * K, K * N, M * N, M * N, K, K, N, N
    };

    Gemm gemm_op;
    size_t workspace_size = Gemm::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // --- Warmup Runs ---
    for (int i = 0; i < warmup_runs; ++i) {
        gemm_op(args, workspace.get());
    }

    // --- Measurement Runs ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < test_runs; ++i) {
        gemm_op(args, workspace.get());
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_runtime_ms = elapsed_ms / test_runs;

    // --- Performance Calculation ---
    double flops = 2.0 * M * N * K;
    double tflops = (flops * 1e-12) / (avg_runtime_ms * 1e-3);

    // --- Report Configuration & Results ---
    std::cout << "--- Benchmark Results ---" << std::endl;
    std::cout << "Avg Runtime: " << avg_runtime_ms << " ms" << std::endl;
    std::cout << "Throughput:  " << tflops << " TFLOPS" << std::endl;
    std::cout << "\n--- Kernel Configuration ---" << std::endl;
    std::cout << "Threadblock Tile: " << Gemm::ThreadblockShape::kM << "x" 
              << Gemm::ThreadblockShape::kN << "x" << Gemm::ThreadblockShape::kK << std::endl;
    std::cout << "Warp Tile:        " << Gemm::WarpShape::kM << "x" 
              << Gemm::WarpShape::kN << "x" << Gemm::WarpShape::kK << std::endl;
    std::cout << "Instruction Shape: " << Gemm::InstructionShape::kM << "x" 
              << Gemm::InstructionShape::kN << "x" << Gemm::InstructionShape::kK << std::endl;
    std::cout << "Stages:           " << Gemm::kStages << std::endl;
    std::cout << "Warp Count:       " << (Gemm::ThreadblockShape::kM / Gemm::WarpShape::kM) * (Gemm::ThreadblockShape::kN / Gemm::WarpShape::kN) << std::endl;

    return 0;
}