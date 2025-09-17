#include "transformer.h"
#include "metal_gpu_simple.h"
#include <random>

// Static member initialization
ExecutionMode Matrix::s_executionMode = ExecutionMode::CPU;

// Matrix operations implementation
Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    
    if (rows == 0 || cols == 0 || other.cols == 0) {
        return Matrix(rows, other.cols);  // Return zero matrix
    }
    
    // Check if we should use GPU acceleration
    if (s_executionMode == ExecutionMode::GPU || 
        (s_executionMode == ExecutionMode::AUTO && shouldUseGPU(rows, other.cols))) {
        return multiply_gpu(other);
    }
    
    Matrix result(rows, other.cols);
    for (size_t i = 0; i < rows && i < data.size(); ++i) {
        for (size_t j = 0; j < other.cols && j < result.data[i].size(); ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols && k < data[i].size() && k < other.data.size(); ++k) {
                if (j < other.data[k].size()) {
                    sum += data[i][k] * other.data[k][j];
                }
            }
            result.data[i][j] = sum;
        }
    }
    return result;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions don't match for addition");
    }
    
    // Check if we should use GPU acceleration
    if (s_executionMode == ExecutionMode::GPU || 
        (s_executionMode == ExecutionMode::AUTO && shouldUseGPU(rows, cols))) {
        return add_gpu(other);
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[j][i] = data[i][j];
        }
    }
    return result;
}

void Matrix::zero() {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = 0.0f;
        }
    }
}

void Matrix::randomize(float std) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = dist(gen);
        }
    }
}

void Matrix::print() const {
    for (size_t i = 0; i < std::min(rows, size_t(5)); ++i) {
        for (size_t j = 0; j < std::min(cols, size_t(5)); ++j) {
            std::cout << data[i][j] << " ";
        }
        if (cols > 5) std::cout << "...";
        std::cout << std::endl;
    }
    if (rows > 5) std::cout << "..." << std::endl;
}

Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols);
}

Matrix Matrix::ones(size_t rows, size_t cols) {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = 1.0f;
        }
    }
    return result;
}

Matrix Matrix::random(size_t rows, size_t cols, float std) {
    Matrix result(rows, cols);
    result.randomize(std);
    return result;
}

// GPU-accelerated matrix operations
Matrix Matrix::multiply_gpu(const Matrix& other) const {
    SimpleMetalGPU& gpu = SimpleMetalGPU::getInstance();
    if (!gpu.isAvailable()) {
        // Fallback to CPU
        return *this * other;
    }
    return gpu.multiply_gpu(*this, other);
}

Matrix Matrix::add_gpu(const Matrix& other) const {
    SimpleMetalGPU& gpu = SimpleMetalGPU::getInstance();
    if (!gpu.isAvailable()) {
        // Fallback to CPU
        return *this + other;
    }
    return gpu.add_gpu(*this, other);
}

Matrix Matrix::transpose_gpu() const {
    // For now, just fallback to CPU (can implement later)
    return transpose();
}

void Matrix::randomize_gpu(float std) {
    // For now, just fallback to CPU (can implement later)
    randomize(std);
}

// Execution mode control
void Matrix::setExecutionMode(ExecutionMode mode) {
    s_executionMode = mode;
    if (mode != ExecutionMode::CPU) {
        // Initialize GPU if needed
        SimpleMetalGPU& gpu = SimpleMetalGPU::getInstance();
        if (!gpu.isAvailable()) {
            gpu.initialize();
        }
    }
}

ExecutionMode Matrix::getExecutionMode() {
    return s_executionMode;
}

bool Matrix::shouldUseGPU(size_t rows, size_t cols) {
    // Use GPU for larger matrices to amortize GPU setup cost
    return (rows * cols) >= (GPU_THRESHOLD * GPU_THRESHOLD);
}

// Activation functions implementation
namespace Activations {
    float relu(float x) {
        return std::max(0.0f, x);
    }
    
    float gelu(float x) {
        // Approximation of GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }
    
    // GPU-accelerated GELU for matrices
    Matrix gelu_matrix(const Matrix& x) {
        // Check if we should use GPU acceleration
        ExecutionMode mode = Matrix::getExecutionMode();
        if (mode == ExecutionMode::GPU || 
            (mode == ExecutionMode::AUTO && Matrix::shouldUseGPU(x.rows, x.cols))) {
            SimpleMetalGPU& gpu = SimpleMetalGPU::getInstance();
            if (gpu.isAvailable()) {
                return gpu.gelu_gpu(x);
            }
        }
        
        // CPU implementation
        Matrix result = x;
        for (size_t i = 0; i < result.rows; ++i) {
            for (size_t j = 0; j < result.cols; ++j) {
                result[i][j] = gelu(result[i][j]);
            }
        }
        return result;
    }
    
    Matrix softmax(const Matrix& x) {
        // Check if we should use GPU acceleration
        ExecutionMode mode = Matrix::getExecutionMode();
        if (mode == ExecutionMode::GPU || 
            (mode == ExecutionMode::AUTO && Matrix::shouldUseGPU(x.rows, x.cols))) {
            SimpleMetalGPU& gpu = SimpleMetalGPU::getInstance();
            if (gpu.isAvailable()) {
                return gpu.softmax_gpu(x);
            }
        }
        
        // CPU implementation
        Matrix result = x;
        
        // Apply softmax to each row (assuming batch dimension)
        for (size_t i = 0; i < result.rows; ++i) {
            // Find max for numerical stability
            float max_val = *std::max_element(result[i].begin(), result[i].end());
            
            // Compute exponentials and sum
            float sum = 0.0f;
            for (size_t j = 0; j < result.cols; ++j) {
                result[i][j] = std::exp(result[i][j] - max_val);
                sum += result[i][j];
            }
            
            // Normalize
            for (size_t j = 0; j < result.cols; ++j) {
                result[i][j] /= sum;
            }
        }
        
        return result;
    }
    
    Matrix layer_norm(const Matrix& x, const Matrix& gamma, const Matrix& beta, float eps) {
        Matrix result = x;
        
        // Apply layer normalization to each row
        for (size_t i = 0; i < result.rows; ++i) {
            // Compute mean
            float mean = 0.0f;
            for (size_t j = 0; j < result.cols; ++j) {
                mean += result[i][j];
            }
            mean /= result.cols;
            
            // Compute variance
            float variance = 0.0f;
            for (size_t j = 0; j < result.cols; ++j) {
                float diff = result[i][j] - mean;
                variance += diff * diff;
            }
            variance /= result.cols;
            
            // Normalize and scale
            float std_dev = std::sqrt(variance + eps);
            for (size_t j = 0; j < result.cols; ++j) {
                result[i][j] = (result[i][j] - mean) / std_dev;
                result[i][j] = result[i][j] * gamma[0][j] + beta[0][j];
            }
        }
        
        return result;
    }
}

// Positional encoding implementation
PositionalEncoding::PositionalEncoding(size_t max_len, size_t model_dim) 
    : encoding(max_len, model_dim), max_seq_len(max_len), d_model(model_dim) {
    
    // Initialize sinusoidal positional encodings
    for (size_t pos = 0; pos < max_len; ++pos) {
        for (size_t i = 0; i < model_dim; ++i) {
            float angle = pos / std::pow(10000.0f, (2.0f * (i / 2)) / model_dim);
            if (i % 2 == 0) {
                encoding[pos][i] = std::sin(angle);
            } else {
                encoding[pos][i] = std::cos(angle);
            }
        }
    }
}

Matrix PositionalEncoding::get_encoding(size_t seq_len) const {
    if (seq_len > max_seq_len) {
        seq_len = max_seq_len;
    }
    
    Matrix result(seq_len, d_model);
    for (size_t i = 0; i < seq_len && i < encoding.rows; ++i) {
        if (encoding.data.size() > i && result.data.size() > i) {
            for (size_t j = 0; j < d_model && j < encoding.cols && j < result.cols; ++j) {
                result[i][j] = encoding[i][j];
            }
        }
    }
    return result;
}
