#include "metal_gpu_simple.h"
#include "transformer.h"
#include <chrono>

#ifdef __APPLE__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#endif

// SimpleMetalGPU implementation
SimpleMetalGPU& SimpleMetalGPU::getInstance() {
    static SimpleMetalGPU instance;
    return instance;
}

bool SimpleMetalGPU::initialize() {
    if (m_initialized) return true;
    
#ifdef __APPLE__
    std::cout << "Initializing Metal GPU acceleration..." << std::endl;
    
    // Get the default Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal: Failed to create default device" << std::endl;
        return false;
    }
    
    // Store device (cast to void*)
    m_device = (__bridge_retained void*)device;
    
    std::cout << "Metal device: " << [[device name] UTF8String] << std::endl;
    
    // Create command queue
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    if (!commandQueue) {
        std::cerr << "Metal: Failed to create command queue" << std::endl;
        return false;
    }
    
    // Store command queue (cast to void*)
    m_commandQueue = (__bridge_retained void*)commandQueue;
    
    m_initialized = true;
    std::cout << "Metal GPU acceleration initialized successfully!" << std::endl;
    
    return true;
#else
    std::cerr << "Metal GPU acceleration not available on this platform" << std::endl;
    return false;
#endif
}

bool SimpleMetalGPU::isAvailable() const {
    return m_initialized;
}

Matrix SimpleMetalGPU::multiply_gpu(const Matrix& a, const Matrix& b) {
#ifdef __APPLE__
    if (!m_initialized) {
        std::cerr << "Metal GPU not initialized" << std::endl;
        return Matrix(0, 0);
    }
    
    
    if (a.cols != b.rows) {
        throw std::invalid_argument("Matrix dimensions don't match for GPU multiplication");
    }
    
    // Cast back to proper types
    id<MTLDevice> device = (__bridge id<MTLDevice>)m_device;
    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)m_commandQueue;
    
    // Use Metal Performance Shaders for optimized matrix multiplication
    NSUInteger M = a.rows;
    NSUInteger N = b.cols;
    NSUInteger K = a.cols;
    
    // Create MPS matrix descriptors
    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M 
                                                                      columns:K 
                                                                     rowBytes:K*sizeof(float) 
                                                                     dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K 
                                                                      columns:N 
                                                                     rowBytes:N*sizeof(float) 
                                                                     dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M 
                                                                      columns:N 
                                                                     rowBytes:N*sizeof(float) 
                                                                     dataType:MPSDataTypeFloat32];
    
    // Create Metal buffers
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    id<MTLBuffer> bufferA = [device newBufferWithLength:sizeA options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithLength:sizeB options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferC = [device newBufferWithLength:sizeC options:MTLResourceStorageModeShared];
    
    // Copy data to GPU buffers
    float* ptrA = (float*)bufferA.contents;
    float* ptrB = (float*)bufferB.contents;
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            ptrA[i * K + j] = a.data[i][j];
        }
    }
    
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < N; ++j) {
            ptrB[i * N + j] = b.data[i][j];
        }
    }
    
    // Create MPS matrices
    MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
    MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
    MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];
    
    // Create and configure matrix multiplication
    MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                       transposeLeft:NO
                                                                      transposeRight:NO
                                                                          resultRows:M
                                                                       resultColumns:N
                                                                    interiorColumns:K
                                                                               alpha:1.0
                                                                                beta:0.0];
    
    // Create command buffer and encode operation
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    [matmul encodeToCommandBuffer:commandBuffer
                       leftMatrix:matrixA 
                      rightMatrix:matrixB 
                     resultMatrix:matrixC];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy result back to CPU
    Matrix result(M, N);
    float* ptrC = (float*)bufferC.contents;
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result.data[i][j] = ptrC[i * N + j];
        }
    }
    
    return result;
#else
    // Direct CPU implementation to avoid recursion
    if (a.cols != b.rows) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    Matrix result(a.rows, b.cols);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < b.cols; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < a.cols; ++k) {
                sum += a.data[i][k] * b.data[k][j];
            }
            result.data[i][j] = sum;
        }
    }
    return result;
#endif
}

Matrix SimpleMetalGPU::add_gpu(const Matrix& a, const Matrix& b) {
#ifdef __APPLE__
    if (!m_initialized) {
        // Direct CPU implementation to avoid recursion
        Matrix result(a.rows, a.cols);
        for (size_t i = 0; i < a.rows; ++i) {
            for (size_t j = 0; j < a.cols; ++j) {
                result.data[i][j] = a.data[i][j] + b.data[i][j];
            }
        }
        return result;
    }
    
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::invalid_argument("Matrix dimensions don't match for GPU addition");
    }
    
    // Cast back to proper types
    id<MTLDevice> device = (__bridge id<MTLDevice>)m_device;
    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)m_commandQueue;
    
    // Use MPS for optimized element-wise addition
    size_t M = a.rows;
    size_t N = a.cols;
    size_t totalElements = M * N;
    
    // Create Metal buffers
    size_t bufferSize = totalElements * sizeof(float);
    id<MTLBuffer> bufferA = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferC = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    
    // Copy data to GPU buffers (row-major layout)
    float* ptrA = (float*)bufferA.contents;
    float* ptrB = (float*)bufferB.contents;
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            ptrA[i * N + j] = a.data[i][j];
            ptrB[i * N + j] = b.data[i][j];
        }
    }
    
    // Use a simple custom compute shader for element-wise addition
    // Create inline compute shader for matrix addition  
    NSString* addShaderSource = @""\
        "#include <metal_stdlib>\n"\
        "using namespace metal;\n"\
        "kernel void matrix_add(device const float* a [[buffer(0)]],\n"\
        "                      device const float* b [[buffer(1)]],\n"\
        "                      device float* result [[buffer(2)]],\n"\
        "                      uint index [[thread_position_in_grid]]) {\n"\
        "    result[index] = a[index] + b[index];\n"\
        "}";
    
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:addShaderSource options:nil error:&error];
    
    if (!library) {
        // Fallback to CPU if shader compilation fails
        Matrix result(a.rows, a.cols);
        for (size_t i = 0; i < a.rows; ++i) {
            for (size_t j = 0; j < a.cols; ++j) {
                result.data[i][j] = a.data[i][j] + b.data[i][j];
            }
        }
        return result;
    }
    
    id<MTLFunction> addFunction = [library newFunctionWithName:@"matrix_add"];
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:addFunction error:&error];
    
    if (!pipelineState) {
        // Fallback to CPU if pipeline creation fails
        Matrix result(a.rows, a.cols);
        for (size_t i = 0; i < a.rows; ++i) {
            for (size_t j = 0; j < a.cols; ++j) {
                result.data[i][j] = a.data[i][j] + b.data[i][j];
            }
        }
        return result;
    }
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipelineState];
    [encoder setBuffer:bufferA offset:0 atIndex:0];
    [encoder setBuffer:bufferB offset:0 atIndex:1];
    [encoder setBuffer:bufferC offset:0 atIndex:2];
    
    // Calculate thread group size
    NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > totalElements) {
        threadGroupSize = totalElements;
    }
    
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);
    MTLSize threadgroupsPerGrid = MTLSizeMake((totalElements + threadGroupSize - 1) / threadGroupSize, 1, 1);
    
    [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy result back to CPU
    Matrix result(M, N);
    float* ptrC = (float*)bufferC.contents;
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result.data[i][j] = ptrC[i * N + j];
        }
    }
    
    return result;
#else
    // Direct CPU implementation to avoid recursion
    Matrix result(a.rows, a.cols);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            result.data[i][j] = a.data[i][j] + b.data[i][j];
        }
    }
    return result;
#endif
}

Matrix SimpleMetalGPU::gelu_gpu(const Matrix& x) {
#ifdef __APPLE__
    if (!m_initialized) {
        // Direct CPU GELU implementation
        Matrix result = x;
        for (size_t i = 0; i < result.rows; ++i) {
            for (size_t j = 0; j < result.cols; ++j) {
                float val = result[i][j];
                result[i][j] = 0.5f * val * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (val + 0.044715f * val * val * val)));
            }
        }
        return result;
    }
    
    // Cast back to proper types
    id<MTLDevice> device = (__bridge id<MTLDevice>)m_device;
    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)m_commandQueue;
    
    size_t M = x.rows;
    size_t N = x.cols;
    size_t totalElements = M * N;
    
    // Create Metal buffers
    size_t bufferSize = totalElements * sizeof(float);
    id<MTLBuffer> inputBuffer = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    
    // Copy input data to GPU
    float* inputPtr = (float*)inputBuffer.contents;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            inputPtr[i * N + j] = x.data[i][j];
        }
    }
    
    // Create compute shader source (inline for simplicity)
    NSString* shaderSource = @""\
        "#include <metal_stdlib>\n"\
        "using namespace metal;\n"\
        "kernel void gelu_kernel(device const float* input [[buffer(0)]],\n"\
        "                       device float* output [[buffer(1)]],\n"\
        "                       uint index [[thread_position_in_grid]]) {\n"\
        "    float x = input[index];\n"\
        "    float sqrt_2_pi = 0.7978845608f;\n"\
        "    float gelu_coeff = 0.044715f;\n"\
        "    float inner = sqrt_2_pi * (x + gelu_coeff * x * x * x);\n"\
        "    float tanh_val = tanh(inner);\n"\
        "    output[index] = 0.5f * x * (1.0f + tanh_val);\n"\
        "}";
    
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
    
    if (!library) {
        // Fallback to CPU if shader compilation fails
        Matrix result = x;
        for (size_t i = 0; i < result.rows; ++i) {
            for (size_t j = 0; j < result.cols; ++j) {
                float val = result[i][j];
                result[i][j] = 0.5f * val * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (val + 0.044715f * val * val * val)));
            }
        }
        return result;
    }
    
    id<MTLFunction> kernelFunction = [library newFunctionWithName:@"gelu_kernel"];
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
    
    if (!pipelineState) {
        // Fallback to CPU if pipeline creation fails
        Matrix result = x;
        for (size_t i = 0; i < result.rows; ++i) {
            for (size_t j = 0; j < result.cols; ++j) {
                float val = result[i][j];
                result[i][j] = 0.5f * val * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (val + 0.044715f * val * val * val)));
            }
        }
        return result;
    }
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipelineState];
    [encoder setBuffer:inputBuffer offset:0 atIndex:0];
    [encoder setBuffer:outputBuffer offset:0 atIndex:1];
    
    // Calculate thread group size
    NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > totalElements) {
        threadGroupSize = totalElements;
    }
    
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);
    MTLSize threadgroupsPerGrid = MTLSizeMake((totalElements + threadGroupSize - 1) / threadGroupSize, 1, 1);
    
    [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy result back to CPU
    Matrix result(M, N);
    float* outputPtr = (float*)outputBuffer.contents;
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result.data[i][j] = outputPtr[i * N + j];
        }
    }
    
    return result;
#else
    // Direct CPU GELU implementation
    Matrix result = x;
    for (size_t i = 0; i < result.rows; ++i) {
        for (size_t j = 0; j < result.cols; ++j) {
            float val = result[i][j];
            result[i][j] = 0.5f * val * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (val + 0.044715f * val * val * val)));
        }
    }
    return result;
#endif
}

// Placeholder for layer_norm_gpu - would be complex to implement efficiently
Matrix SimpleMetalGPU::layer_norm_gpu(const Matrix& x, const Matrix& gamma, const Matrix& beta, float eps) {
    // For now, fallback to CPU implementation as layer norm requires complex reduction operations
    // that would need custom compute shaders with shared memory optimizations
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

Matrix SimpleMetalGPU::softmax_gpu(const Matrix& x) {
#ifdef __APPLE__
    if (!m_initialized) {
        // Direct CPU softmax implementation
        Matrix result = x;
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
    
    // Cast back to proper types
    id<MTLDevice> device = (__bridge id<MTLDevice>)m_device;
    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)m_commandQueue;
    
    size_t M = x.rows;
    size_t N = x.cols;
    
    // Create Metal buffers
    size_t bufferSize = M * N * sizeof(float);
    id<MTLBuffer> inputBuffer = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    
    // Copy input data to GPU
    float* inputPtr = (float*)inputBuffer.contents;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            inputPtr[i * N + j] = x.data[i][j];
        }
    }
    
    // Create MPS matrix descriptors
    MPSMatrixDescriptor* inputDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M 
                                                                          columns:N 
                                                                         rowBytes:N*sizeof(float) 
                                                                         dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* outputDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M 
                                                                           columns:N 
                                                                          rowBytes:N*sizeof(float) 
                                                                          dataType:MPSDataTypeFloat32];
    
    // Create MPS matrices
    MPSMatrix* inputMatrix = [[MPSMatrix alloc] initWithBuffer:inputBuffer descriptor:inputDesc];
    MPSMatrix* outputMatrix = [[MPSMatrix alloc] initWithBuffer:outputBuffer descriptor:outputDesc];
    
    // Create softmax operation
    MPSMatrixSoftMax* softmax = [[MPSMatrixSoftMax alloc] initWithDevice:device];
    
    // Create command buffer and encode operation
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    [softmax encodeToCommandBuffer:commandBuffer
                       inputMatrix:inputMatrix
                      resultMatrix:outputMatrix];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy result back to CPU
    Matrix result(M, N);
    float* outputPtr = (float*)outputBuffer.contents;
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result.data[i][j] = outputPtr[i * N + j];
        }
    }
    
    return result;
#else
    // Direct CPU softmax implementation
    Matrix result = x;
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
#endif
}

void SimpleMetalGPU::printPerformanceInfo() const {
    std::cout << "\n=== Metal GPU Performance Info ===" << std::endl;
    std::cout << "Initialized: " << (m_initialized ? "Yes" : "No") << std::endl;
    
#ifdef __APPLE__
    if (m_device) {
        id<MTLDevice> device = (__bridge id<MTLDevice>)m_device;
        std::cout << "Device: " << [[device name] UTF8String] << std::endl;
        std::cout << "Max threads per threadgroup: " << [device maxThreadsPerThreadgroup].width << std::endl;
    }
#endif
    
    std::cout << "====================================" << std::endl;
}

SimpleMetalGPU::~SimpleMetalGPU() {
#ifdef __APPLE__
    if (m_device) {
        CFRelease(m_device);
    }
    if (m_commandQueue) {
        CFRelease(m_commandQueue);
    }
#endif
}

// MetalUtils implementation
namespace MetalUtils {
    bool isMetalSupported() {
#ifdef __APPLE__
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil;
#else
        return false;
#endif
    }
    
    Timer::Timer() {}
    
    void Timer::start() {
        m_start = std::chrono::high_resolution_clock::now();
    }
    
    double Timer::stop() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start);
        return duration.count() / 1000.0; // Convert to milliseconds
    }
}