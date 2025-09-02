#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

// Helper macro to check for CUDA errors
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

// Tile width for the shared memory version
#define TILE_WIDTH 16

// ============================================================================
// Naive GPU Kernel (reads directly from global memory)
// ============================================================================
__global__ void matrixMulBasic(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Tiled GPU Kernel (uses shared memory to reduce global memory access)
// ============================================================================
__global__ void matrixMulTiled(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Cooperatively load tiles into shared memory
        if (row < M && (t * TILE_WIDTH + tx) < K) {
            ds_A[ty][tx] = A[row * K + (t * TILE_WIDTH + tx)];
        } else {
            ds_A[ty][tx] = 0.0f;
        }

        if ((t * TILE_WIDTH + ty) < K && col < N) {
            ds_B[ty][tx] = B[(t * TILE_WIDTH + ty) * N + col];
        } else {
            ds_B[ty][tx] = 0.0f;
        }
        
        __syncthreads();

        // Perform matrix multiplication on the tiles in shared memory
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += ds_A[ty][i] * ds_B[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// CPU reference implementation for verification
// ============================================================================
void matrixMulCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int K, int N) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// ============================================================================
// Verification function
// ============================================================================
void verifyResult(const std::vector<float>& gpu_result, const std::vector<float>& cpu_result, int M, int N) {
    const float epsilon = 1e-4f;
    bool mismatch = false;
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(gpu_result[i] - cpu_result[i]) > epsilon) {
            std::cerr << "Verification FAILED at index " << i << "! ";
            std::cerr << "GPU result: " << gpu_result[i] << ", CPU result: " << cpu_result[i] << std::endl;
            mismatch = true;
            break;
        }
    }
    if (!mismatch) {
        std::cout << "Verification PASSED!" << std::endl;
    }
}


int main() {
    // Matrix dimensions (A: M x K, B: K x N, C: M x N)
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    std::cout << "Benchmarking Matrix Multiplication" << std::endl;
    std::cout << "Matrix Dimensions: A(" << M << "x" << K << "), B(" << K << "x" << N << ")" << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    // Host memory allocation
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_gpu(M * N);
    std::vector<float> h_C_cpu(M * N);

    // Initialize matrices with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Copy input data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    // CUDA Events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    float milliseconds = 0;

    // --- Benchmark Naive Kernel ---
    std::cout << "Running Naive Kernel..." << std::endl;
    dim3 threadsPerBlockBasic(16, 16);
    dim3 numBlocksBasic((N + threadsPerBlockBasic.x - 1) / threadsPerBlockBasic.x,
                        (M + threadsPerBlockBasic.y - 1) / threadsPerBlockBasic.y);

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    matrixMulBasic<<<numBlocksBasic, threadsPerBlockBasic>>>(d_A, d_B, d_C, M, K, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << "Naive Kernel Time: " << milliseconds << " ms" << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Benchmark Tiled Kernel ---
    std::cout << "\nRunning Tiled (Shared Memory) Kernel..." << std::endl;
    // Clear device memory before running the next kernel
    CHECK_CUDA_ERROR(cudaMemset(d_C, 0, M * N * sizeof(float))); 

    dim3 threadsPerBlockTiled(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocksTiled((N + TILE_WIDTH - 1) / TILE_WIDTH,
                        (M + TILE_WIDTH - 1) / TILE_WIDTH);

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    matrixMulTiled<<<numBlocksTiled, threadsPerBlockTiled>>>(d_A, d_B, d_C, M, K, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << "Tiled Kernel Time: " << milliseconds << " ms" << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // --- Verification ---
    std::cout << "\nVerifying results against CPU calculation..." << std::endl;
    matrixMulCPU(h_A, h_B, h_C_cpu, M, K, N);
    verifyResult(h_C_gpu, h_C_cpu, M, N);

    // Cleanup
    std::cout << "\nCleaning up..." << std::endl;
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return 0;
}
