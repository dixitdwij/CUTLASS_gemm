#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <pybind11/pybind11.h>  

namespace py=pybind11;

// Configuration for benchmarking
#define WARMUP_RUNS 10
#define MEASURED_RUNS 100

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error: " << err << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

template<typename T>
void random_init(T* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        float r = static_cast<float>(rand()) / RAND_MAX;
        data[i] = static_cast<T>(r);
    }
}

template<>
void random_init<__nv_bfloat16>(__nv_bfloat16* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        float r = static_cast<float>(rand()) / RAND_MAX;
        data[i] = __float2bfloat16(r);
    }
}

enum struct DataType { FP32, FP16, BF16 };

std::ostream& operator<<(std::ostream& os, DataType dtype) {
    switch(dtype) {
        case DataType::FP16: os << "fp16"; break;
        case DataType::FP32: os << "fp32"; break;
        case DataType::BF16: os << "bf16"; break;
        default: os << "Unknown"; break;
    }
    return os;
}

const char* dataTypeToString(DataType dtype) {
    switch(dtype) {
        case DataType::FP16: return "fp16";
        case DataType::FP32: return "fp32";
        case DataType::BF16: return "bf16";
        default: return "Unknown";
    }
}

double run_cublas_gemm(int m, int n, int k, DataType dtype);

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " m k n datatype" << std::endl;
        return EXIT_FAILURE;
    }

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);
    
    DataType dtype;
    if (!strncmp(argv[4], "fp32", 4)) dtype = DataType::FP32;
    else if (!strncmp(argv[4], "fp16", 4)) dtype = DataType::FP16;
    else if (!strncmp(argv[4], "bf16", 4)) dtype = DataType::BF16;
    else {
        std::cerr << "Unsupported datatype. Use: fp32, fp16, bf16" << std::endl;
        return EXIT_FAILURE;
    }

    double tflops = run_cublas_gemm(m, n, k, dtype);

    std::cout << "==========================================\n";
    std::cout << "DataType    : " << dtype << "\n";
    std::cout << "Matrix Size : " << m << " x " << n << " x " << k << "\n";
    std::cout << "Runs        : " << WARMUP_RUNS << " warmup + " << MEASURED_RUNS << " measured\n";
    std::cout << "TFLOPS      : " << tflops << " TFLOP/s\n";
    std::cout << "==========================================\n";

    return 0;
}

double run_cublas_gemm(int m, int n, int k, DataType dtype) {
    void *h_A = nullptr, *h_B = nullptr, *h_C = nullptr;
    size_t A_el = m * k, B_el = k * n, C_el = m * n;
    size_t element_size = 0;

    // Containers
    std::vector<float> A_f, B_f, C_f;
    std::vector<half> A_h, B_h, C_h;
    std::vector<__nv_bfloat16> A_bf, B_bf, C_bf;

    // 1. Initialize Host Data
    if (dtype == DataType::FP32) {
        element_size = sizeof(float);
        A_f.resize(A_el); B_f.resize(B_el); C_f.resize(C_el);
        random_init(A_f.data(), A_el);
        random_init(B_f.data(), B_el);
        h_A = A_f.data(); h_B = B_f.data(); h_C = C_f.data();
    } else if (dtype == DataType::FP16) {
        element_size = sizeof(half);
        A_h.resize(A_el); B_h.resize(B_el); C_h.resize(C_el);
        random_init(A_h.data(), A_el);
        random_init(B_h.data(), B_el);
        h_A = A_h.data(); h_B = B_h.data(); h_C = C_h.data();
    } else if (dtype == DataType::BF16) {
        element_size = sizeof(__nv_bfloat16);
        A_bf.resize(A_el); B_bf.resize(B_el); C_bf.resize(C_el);
        random_init(A_bf.data(), A_el);
        random_init(B_bf.data(), B_el);
        h_A = A_bf.data(); h_B = B_bf.data(); h_C = C_bf.data();
    }

    // 2. Allocate Device Memory
    void *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, A_el * element_size));
    CHECK_CUDA(cudaMalloc(&d_B, B_el * element_size));
    CHECK_CUDA(cudaMalloc(&d_C, C_el * element_size));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, A_el * element_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, B_el * element_size, cudaMemcpyHostToDevice));
    
    // No need to Memset C if beta is 0, but good practice to avoid NaN issues in some cases
    CHECK_CUDA(cudaMemset(d_C, 0, C_el * element_size));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 3. Setup Constants for C = A * B
    // Alpha = 1.0, Beta = 0.0 (Overwrite C)
    float alpha_f = 1.0f, beta_f = 0.0f; 
    half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);

    // WARMUP LOOP
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        if (dtype == DataType::FP32) {
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                     &alpha_f, (float*)d_A, m, (float*)d_B, k, &beta_f, (float*)d_C, m));
        } else if (dtype == DataType::FP16) {
            CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                     &alpha_h, (half*)d_A, m, (half*)d_B, k, &beta_h, (half*)d_C, m));
        } else if (dtype == DataType::BF16) {
            CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                      &alpha_f, d_A, CUDA_R_16BF, m, d_B, CUDA_R_16BF, k, &beta_f, d_C, CUDA_R_16BF, m,
                                      CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }
    // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha_f, d_A   , CUDA_R_16BF, m,    d_B, CUDA_R_16BF, k, &beta_f,    d_C, CUDA_R_16BF, m,          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
    // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha  , dB_f32, CUDA_R_16F , N, dA_f32,  CUDA_R_16F, K,   &beta, dC_ref,  CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
    
    // Ensure warmup is done before starting timer
    CHECK_CUDA(cudaDeviceSynchronize());

    // MEASUREMENT LOOP
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < MEASURED_RUNS; ++i) {
        if (dtype == DataType::FP32) {
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                     &alpha_f, (float*)d_A, m, (float*)d_B, k, &beta_f, (float*)d_C, m));
        } else if (dtype == DataType::FP16) {
            CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                     &alpha_h, (half*)d_A, m, (half*)d_B, k, &beta_h, (half*)d_C, m));
        } else if (dtype == DataType::BF16) {
            CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                      &alpha_f, d_A, CUDA_R_16BF, m, d_B, CUDA_R_16BF, k, &beta_f, d_C, CUDA_R_16BF, m,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float totalTimeMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&totalTimeMs, start, stop));

    // Copy result back (just to verify no crashes)
    CHECK_CUDA(cudaMemcpy(h_C, d_C, C_el * element_size, cudaMemcpyDeviceToHost));

    // Calculate stats
    double avgTimeMs = static_cast<double>(totalTimeMs) / MEASURED_RUNS;
    double flops = 2.0f * m * k * n;
    double tflops = flops / (avgTimeMs / 1000.0f) / 1e12; // TFLOP/s

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return tflops;
}

PYBIND11_MODULE(cublass_gemm, m) {
    m.doc() = "cuBLAS GEMM Benchmarking Module";

    m.def("run_cublas_gemm", &run_cublas_gemm, 
          py::arg("m"), py::arg("n"), py::arg("k"), py::arg("dtype"),
          "Run cuBLAS GEMM and return TFLOPS performance.");

    py::enum_<DataType>(m, "DataType")
        .value("FP32", DataType::FP32)
        .value("FP16", DataType::FP16)
        .value("BF16", DataType::BF16)
        .export_values();

    m.def("dataTypeToString", &dataTypeToString, 
          py::arg("dtype"),
          "Convert DataType enum to string representation.");
}