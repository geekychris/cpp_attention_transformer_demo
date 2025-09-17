#pragma once

#include <memory>
#include <vector>
#include <iostream>
#include <chrono>

// Forward declaration of Matrix class
class Matrix;

class SimpleMetalGPU {
public:
    static SimpleMetalGPU& getInstance();
    
    // Initialize Metal GPU resources
    bool initialize();
    
    // Check if Metal is available
    bool isAvailable() const;
    
    // Matrix operations on GPU
    Matrix multiply_gpu(const Matrix& a, const Matrix& b);
    Matrix add_gpu(const Matrix& a, const Matrix& b);
    
    // Activation functions on GPU
    Matrix softmax_gpu(const Matrix& x);
    Matrix gelu_gpu(const Matrix& x);
    Matrix layer_norm_gpu(const Matrix& x, const Matrix& gamma, const Matrix& beta, float eps = 1e-5f);
    
    // Performance info
    void printPerformanceInfo() const;
    
    ~SimpleMetalGPU();

private:
    SimpleMetalGPU() = default;
    
    void* m_device = nullptr;  // Will cast to id<MTLDevice> in implementation
    void* m_commandQueue = nullptr;  // Will cast to id<MTLCommandQueue> in implementation
    
    bool m_initialized = false;
};

// Utility functions
namespace MetalUtils {
    // Check if system supports Metal
    bool isMetalSupported();
    
    // Performance timing utilities
    class Timer {
    public:
        Timer();
        void start();
        double stop(); // Returns elapsed time in milliseconds
        
    private:
        std::chrono::high_resolution_clock::time_point m_start;
    };
}