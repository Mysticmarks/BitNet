#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef BITNET_GPU_FIXTURE_PATH
#error "BITNET_GPU_FIXTURE_PATH compile definition is required"
#endif

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

struct FixtureData {
    std::vector<float> input;
    std::vector<float> weights;
    std::vector<float> expected;
    int rows = 0;
    int cols = 0;
};

static std::vector<std::string> split(const std::string &value, char delimiter) {
    std::vector<std::string> parts;
    std::string current;
    for (char ch : value) {
        if (ch == delimiter) {
            if (!current.empty()) {
                parts.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(ch);
        }
    }
    if (!current.empty()) {
        parts.push_back(current);
    }
    return parts;
}

static std::vector<float> parse_float_list(const std::string &line) {
    const auto separator = line.find('=');
    if (separator == std::string::npos) {
        throw std::runtime_error("Malformed fixture line: " + line);
    }
    std::vector<float> values;
    for (auto &token : split(line.substr(separator + 1), ',')) {
        values.push_back(std::stof(token));
    }
    return values;
}

static FixtureData load_fixture(const std::string &path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Failed to open fixture: " + path);
    }

    FixtureData fixture{};
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        const auto separator = line.find('=');
        if (separator == std::string::npos) {
            continue;
        }
        const std::string key = line.substr(0, separator);
        const std::string value = line.substr(separator + 1);
        if (key == "rows") {
            fixture.rows = std::stoi(value);
        } else if (key == "cols") {
            fixture.cols = std::stoi(value);
        } else if (key == "input") {
            fixture.input = parse_float_list(line);
        } else if (key == "weights") {
            fixture.weights = parse_float_list(line);
        } else if (key == "expected") {
            fixture.expected = parse_float_list(line);
        }
    }

    if (static_cast<int>(fixture.input.size()) != fixture.cols) {
        throw std::runtime_error("Fixture input shape mismatch");
    }
    if (static_cast<int>(fixture.weights.size()) != fixture.rows * fixture.cols) {
        throw std::runtime_error("Fixture weight shape mismatch");
    }
    if (static_cast<int>(fixture.expected.size()) != fixture.rows) {
        throw std::runtime_error("Fixture expected shape mismatch");
    }
    return fixture;
}

int main() {
    FixtureData fixture = load_fixture(BITNET_GPU_FIXTURE_PATH);

    float *d_input = nullptr;
    float *d_weights = nullptr;
    float *d_output = nullptr;

    const int rows = fixture.rows;
    const int cols = fixture.cols;

    check_cuda(cudaMalloc(&d_input, cols * sizeof(float)), "cudaMalloc input");
    check_cuda(cudaMalloc(&d_weights, rows * cols * sizeof(float)), "cudaMalloc weights");
    check_cuda(cudaMalloc(&d_output, rows * sizeof(float)), "cudaMalloc output");

    check_cuda(cudaMemcpy(d_input, fixture.input.data(), cols * sizeof(float), cudaMemcpyHostToDevice), "cpy input");
    check_cuda(cudaMemcpy(d_weights, fixture.weights.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice), "cpy weights");

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
        max_error = std::max<double>(max_error, std::abs(host_output[i] - fixture.expected[i]));
    }

    const double tokens_per_second = static_cast<double>(rows) / std::max(elapsed.count(), 1e-6);
    std::cout << "token_per_second=" << tokens_per_second << "\n";
    std::cout << "latency_ms=" << elapsed.count() * 1e3 << "\n";
    std::cout << "max_error=" << max_error << "\n";

    if (max_error > 1e-4) {
        std::cerr << "GPU max error too high: " << max_error << "\n";
        return 1;
    }
    return 0;
}
