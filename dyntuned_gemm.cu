#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm.h"

// Macros for Static Parameters
#define REQUIRED_MACROS \
    X(TB_M) \
    X(TB_N) \
    X(TB_K) \
    X(W_M)  \
    X(W_N)  \
    X(W_K)  \
    X(INST_M) \
    X(INST_N) \
    X(INST_K) \
    X(STAGES)

#define X(name) [[maybe_unused]] static const int _check_defined_##name = name;

REQUIRED_MACROS

#undef X

// Run Configuration
#define WARMUP_RUNS 5
#define MEASURED_RUNS 10 

// CPU Reference GEMM
template <typename T>
void cpu_gemm(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float accum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = static_cast<float>(A[i * K + k]);
                float b = static_cast<float>(B[k * N + j]);
                accum += a * b;
            }
            
            float c_old = static_cast<float>(C[i * N + j]);
            C[i * N + j] = static_cast<T>(static_cast<float>(alpha) * accum + static_cast<float>(beta) * c_old);
        }
    }
}

// Random Initialization
template <typename T>
void initialize_tensor(T* data, size_t count, float min = -1.0f, float max = 1.0f) {
    std::default_random_engine generator(2024); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> distribution(min, max);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<T>(distribution(generator));
    }
}

enum struct DataType { FP32, FP16, BF16 };

std::ostream& operator<<(std::ostream& os, DataType dtype) {
    switch(dtype) {
        case DataType::FP16: os << "fp16"; break;
        case DataType::FP32: os << "fp32 (tf32)"; break;
        case DataType::BF16: os << "bf16"; break;
        default: os << "Unknown"; break;
    }
    return os;
}

// Core GEMM Kernel
#define FP_32_TOLERANCE 1e-3
#define FP_16_TOLERANCE 5e-3
#define BF_16_TOLERANCE 5e-2
double run_gemm_core(int m, int n, int k) {

    #if defined(FP_32)
        using ElementType = float;
        constexpr DataType dtype = DataType::FP32;
        [[maybe_unused]] double tolerance = FP_32_TOLERANCE; 
    #elif defined(FP_16)
        using ElementType = cutlass::half_t;
        constexpr DataType dtype = DataType::FP16;
        [[maybe_unused]] double tolerance = FP_16_TOLERANCE;
    #elif defined(BF_16)
        using ElementType = cutlass::bfloat16_t;
        constexpr DataType dtype = DataType::BF16;
        [[maybe_unused]] double tolerance = BF_16_TOLERANCE;
    #else
        static_assert(false, "Define -DFP_32, -DFP_16, or -DBF_16");
    #endif

    #if defined(SwizzleIdentity)
        #ifndef SwizzleN
            #error "SwizzleIdentity requires -DSwizzleN=..."
        #endif
        using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleN>;
        std::string swizzle_info = "Identity (N=" + std::to_string(SwizzleN) + ")";
    #elif defined(SwizzleSplitK)
        #ifndef SwizzleN
            #error "SwizzleSplitK requires -DSwizzleN=..."
        #endif
        using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<SwizzleN>;
        std::string swizzle_info = "SplitK Identity (N=" + std::to_string(SwizzleN) + ")";
    #elif defined(SwizzleHorizontal)
        using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle;
        std::string swizzle_info = "Horizontal";
    #else
        #error "Swizzle policy not defined. Use -DSwizzleIdentity, -DSwizzleSplitK, or -DSwizzleHorizontal"
    #endif

    using Layout = cutlass::layout::RowMajor;
    using ElementAccumulator = float;
    
    using ShapeOp = cutlass::gemm::GemmShape<INST_M, INST_N, INST_K>;
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
        SwizzleThreadBlock,
        STAGES
    >;

    // Memory Allocation (Host & Device)
    size_t count_A = size_t(m) * k; 
    size_t count_B = size_t(k) * n; 
    size_t count_C = size_t(m) * n; 
    size_t size_A = count_A * sizeof(ElementType);
    size_t size_B = count_B * sizeof(ElementType);
    size_t size_C = count_C * sizeof(ElementType);

    // Device pointers
    ElementType *dev_A, *dev_B, *dev_C;
    cudaMalloc(&dev_A, size_A);
    cudaMalloc(&dev_B, size_B);
    cudaMalloc(&dev_C, size_C);

    // Host pointers
    std::vector<ElementType> host_A(count_A);
    std::vector<ElementType> host_B(count_B);
    std::vector<ElementType> host_C(count_C); // Stores GPU result
    std::vector<ElementType> host_Ref(count_C); // Stores CPU Reference

    // Initialization
    initialize_tensor(host_A.data(), count_A);
    initialize_tensor(host_B.data(), count_B);
    // 0 initialise C
    std::fill(host_C.begin(), host_C.end(), static_cast<ElementType>(0.0f));
    std::fill(host_Ref.begin(), host_Ref.end(), static_cast<ElementType>(0.0f));

    // Copy Host -> Device
    cudaMemcpy(dev_A, host_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_B.data(), size_B, cudaMemcpyHostToDevice);
    cudaMemset(dev_C, 0, size_C); // Reset device C

    // Setup Arguments
    // Alpha = 1.0, Beta = 0.0 (Overwrite C)
    typename Gemm::Arguments arguments(
        {m, n, k}, {dev_A, k}, {dev_B, n}, {dev_C, n}, {dev_C, n}, {1.0f, 0.0f}
    );

    Gemm gemm_op;
    size_t ws_size = Gemm::get_workspace_size(arguments);
    void *ws = nullptr;
    if (ws_size > 0) cudaMalloc(&ws, ws_size);

    if(gemm_op.initialize(arguments, ws) != cutlass::Status::kSuccess) {
        std::cerr << "Failed to initialize CUTLASS kernel." << std::endl;
        exit(1);
    }

    // Performance measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Warmup
    for(int i=0; i<WARMUP_RUNS; ++i) {
        cutlass::Status status = gemm_op(arguments, ws);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS Execution failed at warmup iter " << i << ": " 
                      << cutlass::cutlassGetStatusString(status) << std::endl;
            exit(-1);
        }
    }
    cudaDeviceSynchronize();

    // Measurement
    cudaEventRecord(start);
    for(int i=0; i<MEASURED_RUNS; ++i) {
        cutlass::Status status = gemm_op(arguments, ws);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS Execution failed at measurement iter " << i << ": " 
                      << cutlass::cutlassGetStatusString(status) << std::endl;
            exit(-1);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double tflops = (2.0 * m * n * k) / ((ms/MEASURED_RUNS) / 1000.0) / 1e12;

    std::cout << "==========================================\n";
    std::cout << "DataType    : " << dtype << "\n";
    std::cout << "Matrix Size : " << m << " x " << n << " x " << k << "\n";
    std::cout << "  [Config] TB: <" << ShapeTB::kM << "," << ShapeTB::kN << "," << ShapeTB::kK << ">"
              << " Warp: <" << ShapeWarp::kM << "," << ShapeWarp::kN << "," << ShapeWarp::kK << ">"
              << " InstOp: <" << ShapeOp::kM << "," << ShapeOp::kN << "," << ShapeOp::kK << ">"
              << " Swizzle: " << swizzle_info 
              << " Stages: " << STAGES << "\n";
    std::cout << "Runs        : " << WARMUP_RUNS << " warmup + " << MEASURED_RUNS << " measured\n";
    std::cout << "  [Result] Time: " << std::fixed << std::setprecision(3) 
              << ms/MEASURED_RUNS << " ms | "
              << "TFLOPs: " << tflops << std::endl;
    std::cout << "==========================================\n";

    // Verification
    #ifdef VERIFY 
        std::cout << "  [Verify] Running CPU reference check... ";
        std::cout.flush();
        
        // Copy GPU result -> Host
        cudaMemcpy(host_C.data(), dev_C, size_C, cudaMemcpyDeviceToHost);

        // Calculate CPU Reference
        cpu_gemm(m, n, k, 
                 static_cast<ElementType>(1.0f), 
                 host_A.data(), host_B.data(), 
                 static_cast<ElementType>(0.0f), 
                 host_Ref.data());

        // Compare
        double max_err = 0.0;
        for (size_t i = 0; i < count_C; ++i) {
            float gpu_val = static_cast<float>(host_C[i]);
            float ref_val = static_cast<float>(host_Ref[i]);
            float diff = std::abs(gpu_val - ref_val);
            float rel_err = diff / (std::abs(ref_val) + 0.0001f); // Avoid div by zero
            if (rel_err > max_err) max_err = rel_err;
        }

        if (max_err < tolerance) {
            std::cout << "PASS (Max Rel Err: " << max_err << ")" << std::endl;
        } else {
            std::cout << "FAIL (Max Rel Err: " << max_err << " > " << tolerance << ")" << std::endl;
        }
    #endif

    cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C); 
    if(ws) cudaFree(ws);

    return tflops;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K> [verify=0/1]" << std::endl;
        return -1;
    }

    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    int k = std::atoi(argv[3]);

    run_gemm_core(m, n, k);

    return 0;
}