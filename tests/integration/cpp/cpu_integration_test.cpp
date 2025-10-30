#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace minimal_ggml {

struct QuantizedMatrix {
    std::vector<std::uint8_t> weights;  // two bit packed values stored high bits first
    std::vector<float> scales;
    std::size_t rows;
    std::size_t cols;  // logical columns before packing
};

static std::uint8_t get_qvalue(const QuantizedMatrix &mat, std::size_t row, std::size_t col) {
    const std::size_t packed_index = row * ((mat.cols + 3) / 4) + (col / 4);
    const std::size_t shift = 6 - 2 * (col % 4);
    return (mat.weights[packed_index] >> shift) & 0x03u;
}

static float dequantize(std::uint8_t qvalue) {
    // Map two bit values to {-1, 0, 1, 2} to mimic signed ternary like behaviour.
    switch (qvalue) {
        case 0: return -1.0f;
        case 1: return 0.0f;
        case 2: return 1.0f;
        default: return 2.0f;
    }
}

static std::vector<float> infer(const QuantizedMatrix &mat, const std::vector<float> &input) {
    if (input.size() != mat.cols) {
        throw std::invalid_argument("input dimensionality mismatch");
    }
    std::vector<float> output(mat.rows, 0.0f);
    for (std::size_t r = 0; r < mat.rows; ++r) {
        float scale = mat.scales[r];
        for (std::size_t c = 0; c < mat.cols; ++c) {
            const std::uint8_t qvalue = get_qvalue(mat, r, c);
            output[r] += dequantize(qvalue) * input[c] * scale;
        }
        output[r] = std::tanh(output[r]);
    }
    return output;
}

}  // namespace minimal_ggml

int main() {
    using namespace minimal_ggml;

    const std::vector<float> input = {0.25f, -0.75f, 0.5f, 1.0f};

    QuantizedMatrix mat;
    mat.rows = 3;
    mat.cols = input.size();
    mat.scales = {0.75f, 0.5f, 0.9f};
    mat.weights = {
        0b00011011, // qvalues [0,1,2,3]
        0b11100100, // qvalues [3,2,1,0]
        0b01010101, // qvalues [1,1,1,1]
    };

    const auto start = std::chrono::high_resolution_clock::now();
    auto output = infer(mat, input);
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;

    const std::array<float, 3> expected = {
        std::tanh((-1.0f * 0.25f + 0.0f * -0.75f + 1.0f * 0.5f + 2.0f * 1.0f) * 0.75f),
        std::tanh((2.0f * 0.25f + 1.0f * -0.75f + 0.0f * 0.5f + -1.0f * 1.0f) * 0.5f),
        std::tanh((0.0f * 0.25f + 0.0f * -0.75f + 0.0f * 0.5f + 0.0f * 1.0f) * 0.9f)
    };

    double max_error = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        max_error = std::max<double>(max_error, std::abs(output[i] - expected[i]));
    }

    const double tokens_per_second = static_cast<double>(mat.rows) / std::max(elapsed.count(), 1e-6);
    std::cout << "token_per_second=" << tokens_per_second << "\n";
    std::cout << "latency_ms=" << elapsed.count() * 1e3 << "\n";

    if (max_error > 1e-6) {
        std::cerr << "Max error exceeded tolerance: " << max_error << "\n";
        return 1;
    }
    return 0;
}
