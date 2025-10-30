#include <array>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

__global__ void matvec_kernel(const float* __restrict__ weights,
                              const float* __restrict__ input,
                              float* __restrict__ output,
                              int rows,
                              int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    float acc = 0.0f;
    for (int c = 0; c < cols; ++c) {
        acc += weights[row * cols + c] * input[c];
    }
    output[row] = tanhf(acc);
}

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

int main() {
    const std::vector<float> host_input = {0.5f, -1.25f, 0.75f};
    const std::vector<float> host_weights = {
        -0.5f, 1.0f, 0.25f,
        0.75f, -0.25f, -0.5f,
        0.5f, 0.5f, 0.5f
    };
    const std::array<float, 3> expected = {
        std::tanh(-0.5f * 0.5f + 1.0f * -1.25f + 0.25f * 0.75f),
        std::tanh(0.75f * 0.5f + -0.25f * -1.25f + -0.5f * 0.75f),
        std::tanh(0.5f * 0.5f + 0.5f * -1.25f + 0.5f * 0.75f)
    };

    float *d_input = nullptr;
    float *d_weights = nullptr;
    float *d_output = nullptr;

    const int rows = 3;
    const int cols = 3;

    check_cuda(cudaMalloc(&d_input, cols * sizeof(float)), "cudaMalloc input");
    check_cuda(cudaMalloc(&d_weights, rows * cols * sizeof(float)), "cudaMalloc weights");
    check_cuda(cudaMalloc(&d_output, rows * sizeof(float)), "cudaMalloc output");

    check_cuda(cudaMemcpy(d_input, host_input.data(), cols * sizeof(float), cudaMemcpyHostToDevice), "cpy input");
    check_cuda(cudaMemcpy(d_weights, host_weights.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice), "cpy weights");

    const dim3 block(32);
    const dim3 grid((rows + block.x - 1) / block.x);

    const auto start = std::chrono::high_resolution_clock::now();
    matvec_kernel<<<grid, block>>>(d_weights, d_input, d_output, rows, cols);
    check_cuda(cudaDeviceSynchronize(), "kernel sync");
    const auto end = std::chrono::high_resolution_clock::now();

    std::vector<float> host_output(rows, 0.0f);
    check_cuda(cudaMemcpy(host_output.data(), d_output, rows * sizeof(float), cudaMemcpyDeviceToHost), "cpy output");

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);

    const std::chrono::duration<double> elapsed = end - start;
    double max_error = 0.0;
    for (int i = 0; i < rows; ++i) {
        max_error = std::max<double>(max_error, std::abs(host_output[i] - expected[i]));
    }

    const double tokens_per_second = static_cast<double>(rows) / std::max(elapsed.count(), 1e-6);
    std::cout << "token_per_second=" << tokens_per_second << "\n";
    std::cout << "latency_ms=" << elapsed.count() * 1e3 << "\n";

    if (max_error > 1e-4) {
        std::cerr << "GPU max error too high: " << max_error << "\n";
        return 1;
    }
    return 0;
}
