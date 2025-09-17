#include "metal_gpu.h"
#include "transformer.h"
#include <chrono>
#include <random>

#ifdef __APPLE__
#include <Metal/Metal.hpp>
#include <MetalPerformanceShaders/MetalPerformanceShaders.hpp>
#include <Foundation/Foundation.hpp>

// Helper to convert NSString to std::string
std::string nsstring_to_string(NS::String* nsstr) {
    if (!nsstr) return "";
    const char* cstr = nsstr->utf8String();
    return std::string(cstr ? cstr : "");
}
#endif

// MetalGPU implementation
MetalGPU& MetalGPU::getInstance() {
    static MetalGPU instance;
    return instance;
}

bool MetalGPU::initialize() {
    if (m_initialized) return true;
    
#ifdef __APPLE__
    std::cout << "Initializing Metal GPU acceleration..." << std::endl;
    
    // Get the default Metal device
    m_device = MTL::CreateSystemDefaultDevice();
    if (!m_device) {
        std::cerr << "Metal: Failed to create default device" << std::endl;
        return false;
    }
    
    std::cout << "Metal device: " << nsstring_to_string(m_device->name()) << std::endl;
    
    // Create command queue
    m_commandQueue = m_device->newCommandQueue();
    if (!m_commandQueue) {
        std::cerr << "Metal: Failed to create command queue" << std::endl;
        return false;
    }
    
    // Load the Metal library
    NS::Error* error = nullptr;
    NS::String* libraryPath = NS::String::string("matrix_ops.metallib", NS::ASCIIStringEncoding);
    
    // Try to load precompiled library first
    m_library = m_device->newLibrary(libraryPath, &error);
    
    // If that fails, try to compile from source
    if (!m_library) {
        std::cout << "Compiling Metal shaders from source..." << std::endl;
        
        // Read the metal source file
        std::ifstream metalFile("matrix_ops.metal");
        if (!metalFile.is_open()) {
            std::cerr << "Metal: Could not open matrix_ops.metal" << std::endl;
            return false;
        }
        
        std::string metalSource((std::istreambuf_iterator<char>(metalFile)),
                               std::istreambuf_iterator<char>());
        metalFile.close();
        
        NS::String* sourceString = NS::String::string(metalSource.c_str(), NS::UTF8StringEncoding);
        MTL::CompileOptions* compileOptions = MTL::CompileOptions::alloc()->init();
        
        m_library = m_device->newLibrary(sourceString, compileOptions, &error);
        compileOptions->release();
        
        if (!m_library) {
            std::cerr << "Metal: Failed to compile library" << std::endl;
            if (error) {
                std::cerr << "Error: " << nsstring_to_string(error->localizedDescription()) << std::endl;
            }
            return false;
        }
    }
    
    // Create compute pipelines
    if (!createComputePipelines()) {
        std::cerr << "Metal: Failed to create compute pipelines" << std::endl;
        return false;
    }
    
    m_initialized = true;
    std::cout << "Metal GPU acceleration initialized successfully!" << std::endl;
    
    return true;
#else
    std::cerr << "Metal GPU acceleration not available on this platform" << std::endl;
    return false;
#endif
}

#ifdef __APPLE__
bool MetalGPU::createComputePipelines() {
    NS::Error* error = nullptr;
    
    // Matrix addition pipeline
    MTL::Function* addFunction = m_library->newFunction(NS::String::string("matrix_add", NS::ASCIIStringEncoding));
    if (addFunction) {
        m_addPipeline = m_device->newComputePipelineState(addFunction, &error);
        addFunction->release();
        if (!m_addPipeline) {
            std::cerr << "Failed to create matrix_add pipeline: " << nsstring_to_string(error->localizedDescription()) << std::endl;
            return false;
        }
    }
    
    // Matrix transpose pipeline
    MTL::Function* transposeFunction = m_library->newFunction(NS::String::string("matrix_transpose", NS::ASCIIStringEncoding));
    if (transposeFunction) {
        m_transposePipeline = m_device->newComputePipelineState(transposeFunction, &error);
        transposeFunction->release();
        if (!m_transposePipeline) {
            std::cerr << "Failed to create matrix_transpose pipeline" << std::endl;
            return false;
        }
    }
    
    // Matrix randomization pipeline
    MTL::Function* randomizeFunction = m_library->newFunction(NS::String::string("matrix_randomize", NS::ASCIIStringEncoding));
    if (randomizeFunction) {
        m_randomizePipeline = m_device->newComputePipelineState(randomizeFunction, &error);
        randomizeFunction->release();
        if (!m_randomizePipeline) {
            std::cerr << "Failed to create matrix_randomize pipeline" << std::endl;
            return false;
        }
    }
    
    std::cout << "Metal compute pipelines created successfully" << std::endl;
    return true;
}
#endif

bool MetalGPU::isAvailable() const {
    return m_initialized;
}

Matrix MetalGPU::multiply_gpu(const Matrix& a, const Matrix& b) {
#ifdef __APPLE__
    if (!m_initialized) {
        std::cerr << "Metal GPU not initialized" << std::endl;
        return Matrix(0, 0);
    }
    
    if (a.cols != b.rows) {
        throw std::invalid_argument("Matrix dimensions don't match for GPU multiplication");
    }
    
    // Use Metal Performance Shaders for optimized matrix multiplication
    size_t M = a.rows;
    size_t N = b.cols;
    size_t K = a.cols;
    
    // Create MPS matrix descriptors
    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];
    
    // Create Metal buffers
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    MTL::Buffer* bufferA = m_device->newBuffer(sizeA, MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferB = m_device->newBuffer(sizeB, MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferC = m_device->newBuffer(sizeC, MTL::ResourceStorageModeShared);
    
    // Copy data to GPU buffers
    float* ptrA = (float*)bufferA->contents();
    float* ptrB = (float*)bufferB->contents();
    
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
    MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)bufferA descriptor:descA];
    MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)bufferB descriptor:descB];
    MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)bufferC descriptor:descC];
    
    // Create and configure matrix multiplication
    MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc] initWithDevice:(__bridge id<MTLDevice>)m_device
                                                                          transposeLeft:NO
                                                                         transposeRight:NO
                                                                             resultRows:M
                                                                          resultColumns:N
                                                                       interiorColumns:K
                                                                                  alpha:1.0
                                                                                   beta:0.0];
    
    // Create command buffer and encode operation
    MTL::CommandBuffer* commandBuffer = m_commandQueue->commandBuffer();
    [matmul encodeToCommandBuffer:(__bridge id<MTLCommandBuffer>)commandBuffer
                       leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    // Copy result back to CPU
    Matrix result(M, N);
    float* ptrC = (float*)bufferC->contents();
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result.data[i][j] = ptrC[i * N + j];
        }
    }
    
    // Clean up
    [matrixA release];
    [matrixB release];
    [matrixC release];
    [matmul release];
    bufferA->release();
    bufferB->release();
    bufferC->release();
    
    return result;
#else
    // Fallback to CPU implementation
    return a * b;
#endif
}

Matrix MetalGPU::add_gpu(const Matrix& a, const Matrix& b) {
#ifdef __APPLE__
    if (!m_initialized || !m_addPipeline) {
        return a + b; // Fallback to CPU
    }
    
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::invalid_argument("Matrix dimensions don't match for GPU addition");
    }
    
    size_t totalElements = a.rows * a.cols;
    size_t bufferSize = totalElements * sizeof(float);
    
    // Create Metal buffers
    MTL::Buffer* bufferA = m_device->newBuffer(bufferSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferB = m_device->newBuffer(bufferSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferResult = m_device->newBuffer(bufferSize, MTL::ResourceStorageModeShared);
    
    // Copy input data
    float* ptrA = (float*)bufferA->contents();
    float* ptrB = (float*)bufferB->contents();
    
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            size_t idx = i * a.cols + j;
            ptrA[idx] = a.data[i][j];
            ptrB[idx] = b.data[i][j];
        }
    }
    
    // Create command buffer and encoder
    MTL::CommandBuffer* commandBuffer = m_commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    
    encoder->setComputePipelineState(m_addPipeline);
    encoder->setBuffer(bufferA, 0, 0);
    encoder->setBuffer(bufferB, 0, 1);
    encoder->setBuffer(bufferResult, 0, 2);
    
    uint32_t rows = static_cast<uint32_t>(a.rows);
    uint32_t cols = static_cast<uint32_t>(a.cols);
    encoder->setBytes(&rows, sizeof(uint32_t), 3);
    encoder->setBytes(&cols, sizeof(uint32_t), 4);
    
    // Calculate thread group sizes
    MTL::Size threadGroupSize = MTL::Size(16, 16, 1);
    MTL::Size gridSize = MTL::Size((cols + 15) / 16, (rows + 15) / 16, 1);
    
    encoder->dispatchThreadgroups(gridSize, threadGroupSize);
    encoder->endEncoding();
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    // Copy result back
    Matrix result(a.rows, a.cols);
    float* ptrResult = (float*)bufferResult->contents();
    
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            result.data[i][j] = ptrResult[i * a.cols + j];
        }
    }
    
    // Clean up
    bufferA->release();
    bufferB->release();
    bufferResult->release();
    
    return result;
#else
    return a + b;
#endif
}

Matrix MetalGPU::transpose_gpu(const Matrix& a) {
#ifdef __APPLE__
    if (!m_initialized || !m_transposePipeline) {
        return a.transpose(); // Fallback to CPU
    }
    
    size_t inputSize = a.rows * a.cols * sizeof(float);
    size_t outputSize = a.cols * a.rows * sizeof(float);
    
    MTL::Buffer* inputBuffer = m_device->newBuffer(inputSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* outputBuffer = m_device->newBuffer(outputSize, MTL::ResourceStorageModeShared);
    
    // Copy input data
    float* inputPtr = (float*)inputBuffer->contents();
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            inputPtr[i * a.cols + j] = a.data[i][j];
        }
    }
    
    // Create command buffer and encoder
    MTL::CommandBuffer* commandBuffer = m_commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    
    encoder->setComputePipelineState(m_transposePipeline);
    encoder->setBuffer(inputBuffer, 0, 0);
    encoder->setBuffer(outputBuffer, 0, 1);
    
    uint32_t rows = static_cast<uint32_t>(a.rows);
    uint32_t cols = static_cast<uint32_t>(a.cols);
    encoder->setBytes(&rows, sizeof(uint32_t), 2);
    encoder->setBytes(&cols, sizeof(uint32_t), 3);
    
    MTL::Size threadGroupSize = MTL::Size(16, 16, 1);
    MTL::Size gridSize = MTL::Size((cols + 15) / 16, (rows + 15) / 16, 1);
    
    encoder->dispatchThreadgroups(gridSize, threadGroupSize);
    encoder->endEncoding();
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    // Copy result back
    Matrix result(a.cols, a.rows);
    float* outputPtr = (float*)outputBuffer->contents();
    
    for (size_t i = 0; i < result.rows; ++i) {
        for (size_t j = 0; j < result.cols; ++j) {
            result.data[i][j] = outputPtr[i * result.cols + j];
        }
    }
    
    // Clean up
    inputBuffer->release();
    outputBuffer->release();
    
    return result;
#else
    return a.transpose();
#endif
}

void MetalGPU::randomize_gpu(Matrix& matrix, float std_dev) {
#ifdef __APPLE__
    if (!m_initialized || !m_randomizePipeline) {
        matrix.randomize(std_dev); // Fallback to CPU
        return;
    }
    
    size_t totalElements = matrix.rows * matrix.cols;
    size_t bufferSize = totalElements * sizeof(float);
    
    // Generate random values on CPU first
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    std::vector<float> randomValues(totalElements);
    for (size_t i = 0; i < totalElements; ++i) {
        randomValues[i] = dis(gen);
    }
    
    MTL::Buffer* matrixBuffer = m_device->newBuffer(bufferSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* randomBuffer = m_device->newBuffer(bufferSize, MTL::ResourceStorageModeShared);
    
    // Copy random values to GPU
    memcpy(randomBuffer->contents(), randomValues.data(), bufferSize);
    
    // Create command buffer and encoder
    MTL::CommandBuffer* commandBuffer = m_commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    
    encoder->setComputePipelineState(m_randomizePipeline);
    encoder->setBuffer(matrixBuffer, 0, 0);
    encoder->setBuffer(randomBuffer, 0, 1);
    
    uint32_t size = static_cast<uint32_t>(totalElements);
    encoder->setBytes(&size, sizeof(uint32_t), 2);
    encoder->setBytes(&std_dev, sizeof(float), 3);
    
    MTL::Size threadGroupSize = MTL::Size(64, 1, 1);
    MTL::Size gridSize = MTL::Size((totalElements + 63) / 64, 1, 1);
    
    encoder->dispatchThreadgroups(gridSize, threadGroupSize);
    encoder->endEncoding();
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    // Copy result back
    float* resultPtr = (float*)matrixBuffer->contents();
    for (size_t i = 0; i < matrix.rows; ++i) {
        for (size_t j = 0; j < matrix.cols; ++j) {
            matrix.data[i][j] = resultPtr[i * matrix.cols + j];
        }
    }
    
    // Clean up
    matrixBuffer->release();
    randomBuffer->release();
#else
    matrix.randomize(std_dev);
#endif
}

void MetalGPU::waitForCompletion() {
#ifdef __APPLE__
    // This is handled by waitUntilCompleted() calls in individual operations
#endif
}

void MetalGPU::printPerformanceInfo() const {
    std::cout << "\n=== Metal GPU Performance Info ===" << std::endl;
    std::cout << "Initialized: " << (m_initialized ? "Yes" : "No") << std::endl;
    
#ifdef __APPLE__
    if (m_device) {
        std::cout << "Device: " << nsstring_to_string(m_device->name()) << std::endl;
        std::cout << "Max threads per threadgroup: " << m_device->maxThreadsPerThreadgroup().width << std::endl;
        std::cout << "Memory allocated: " << (m_totalMemoryAllocated / 1024.0 / 1024.0) << " MB" << std::endl;
    }
#endif
    
    std::cout << "====================================" << std::endl;
}

MetalGPU::~MetalGPU() {
#ifdef __APPLE__
    if (m_addPipeline) m_addPipeline->release();
    if (m_transposePipeline) m_transposePipeline->release();
    if (m_randomizePipeline) m_randomizePipeline->release();
    if (m_library) m_library->release();
    if (m_commandQueue) m_commandQueue->release();
    if (m_device) m_device->release();
#endif
}

// MetalUtils implementation
namespace MetalUtils {
    bool isMetalSupported() {
#ifdef __APPLE__
        MTL::Device* device = MTL::CreateSystemDefaultDevice();
        if (device) {
            device->release();
            return true;
        }
        return false;
#else
        return false;
#endif
    }
    
    std::pair<size_t, size_t> getOptimalThreadGroupSize(size_t rows, size_t cols) {
        // Choose thread group size based on matrix dimensions
        size_t groupX = std::min(cols, size_t(16));
        size_t groupY = std::min(rows, size_t(16));
        
        return std::make_pair(groupX, groupY);
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