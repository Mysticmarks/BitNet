#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef BITNET_CPU_FIXTURE_PATH
#error "BITNET_CPU_FIXTURE_PATH compile definition is required"
#endif

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

struct FixtureData {
    QuantizedMatrix matrix;
    std::vector<float> input;
    std::vector<float> expected;
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

template <typename T>
static std::vector<T> parse_list(const std::string &line) {
    const auto separator = line.find('=');
    if (separator == std::string::npos) {
        throw std::runtime_error("Malformed fixture line: " + line);
    }
    std::vector<T> values;
    for (auto &token : split(line.substr(separator + 1), ',')) {
        if constexpr (std::is_same_v<T, std::uint8_t>) {
            values.push_back(static_cast<std::uint8_t>(std::stoi(token)));
        } else {
            values.push_back(static_cast<T>(std::stof(token)));
        }
    }
    return values;
}

static FixtureData load_fixture(const std::string &path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Failed to open fixture: " + path);
    }

    FixtureData data{};
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
            data.matrix.rows = static_cast<std::size_t>(std::stoul(value));
        } else if (key == "cols") {
            data.matrix.cols = static_cast<std::size_t>(std::stoul(value));
        } else if (key == "input") {
            data.input = parse_list<float>(line);
        } else if (key == "scales") {
            data.matrix.scales = parse_list<float>(line);
        } else if (key == "weights") {
            data.matrix.weights = parse_list<std::uint8_t>(line);
        } else if (key == "expected") {
            data.expected = parse_list<float>(line);
        }
    }

    if (data.input.size() != data.matrix.cols) {
        throw std::runtime_error("Fixture input shape mismatch");
    }
    if (data.matrix.scales.size() != data.matrix.rows) {
        throw std::runtime_error("Fixture scale shape mismatch");
    }
    if (data.expected.size() != data.matrix.rows) {
        throw std::runtime_error("Fixture expected shape mismatch");
    }
    const auto expected_weights = data.matrix.rows * ((data.matrix.cols + 3) / 4);
    if (data.matrix.weights.size() != expected_weights) {
        throw std::runtime_error("Fixture weight shape mismatch");
    }

    return data;
}

}  // namespace minimal_ggml

int main() {
    using namespace minimal_ggml;

    FixtureData fixture = load_fixture(BITNET_CPU_FIXTURE_PATH);

    const auto start = std::chrono::high_resolution_clock::now();
    auto output = infer(fixture.matrix, fixture.input);
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;

    double max_error = 0.0;
    for (std::size_t i = 0; i < fixture.expected.size(); ++i) {
        max_error = std::max<double>(max_error, std::abs(output[i] - fixture.expected[i]));
    }

    const double tokens_per_second = static_cast<double>(fixture.matrix.rows) / std::max(elapsed.count(), 1e-6);
    std::cout << "token_per_second=" << tokens_per_second << "\n";
    std::cout << "latency_ms=" << elapsed.count() * 1e3 << "\n";
    std::cout << "max_error=" << max_error << "\n";

    if (max_error > 1e-6) {
        std::cerr << "Max error exceeded tolerance: " << max_error << "\n";
        return 1;
    }
    return 0;
}
