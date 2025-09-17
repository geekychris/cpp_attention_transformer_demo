#pragma once

#ifdef __APPLE__
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Foundation/Foundation.h>
#endif

#include <memory>
#include <vector>
#include <iostream>

// Forward declaration of Matrix class
class Matrix;

class MetalGPU {
public:
    static MetalGPU& getInstance();
    
    // Initialize Metal GPU resources
    bool initialize();
    
    // Check if Metal is available
    bool isAvailable() const;
    
    // Matrix operations on GPU
    Matrix multiply_gpu(const Matrix& a, const Matrix& b);
    Matrix add_gpu(const Matrix& a, const Matrix& b);
    Matrix transpose_gpu(const Matrix& a);
    void randomize_gpu(Matrix& matrix, float std_dev);
    
    // Memory management
    void* allocateBuffer(size_t size);
    void deallocateBuffer(void* buffer);
    
    // Synchronization
    void waitForCompletion();
    
    // Performance info
    void printPerformanceInfo() const;
    
    ~MetalGPU();

private:
    MetalGPU() = default;
    
#ifdef __APPLE__
    id<MTLDevice> m_device;
    id<MTLCommandQueue> m_commandQueue;
    id<MTLLibrary> m_library;
    
    // Metal Performance Shaders
    MPSMatrixMultiplication* m_matmul;
    
    // Compute pipelines for custom operations
    id<MTLComputePipelineState> m_addPipeline;
    id<MTLComputePipelineState> m_transposePipeline;
    id<MTLComputePipelineState> m_randomizePipeline;
    
    bool createComputePipelines();
#endif
    
    bool m_initialized = false;
    size_t m_totalMemoryAllocated = 0;
};

// GPU-accelerated buffer class
class MetalBuffer {
public:
    MetalBuffer(size_t size);
    ~MetalBuffer();
    
    void* getData() const { return m_data; }
    size_t getSize() const { return m_size; }
    
    // Copy data to/from GPU
    void copyFromHost(const void* hostData, size_t size);
    void copyToHost(void* hostData, size_t size) const;

private:
#ifdef __APPLE__
    id<MTLBuffer> m_buffer;
#endif
    void* m_data = nullptr;
    size_t m_size = 0;
};

// Utility functions
namespace MetalUtils {
    // Check if system supports Metal
    bool isMetalSupported();
    
    // Get optimal thread group size for given matrix dimensions
    std::pair<size_t, size_t> getOptimalThreadGroupSize(size_t rows, size_t cols);
    
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