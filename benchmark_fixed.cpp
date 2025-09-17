#include "training.h"
#include "metal_gpu_simple.h"
#include <iostream>
#include <chrono>
#include <iomanip>

namespace BenchmarkUtils {
    class Timer {
    public:
        void start() {
            m_start = std::chrono::high_resolution_clock::now();
        }
        
        double stop() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start);
            return duration.count() / 1000.0; // Convert to milliseconds
        }
        
    private:
        std::chrono::high_resolution_clock::time_point m_start;
    };
}

void benchmarkMatrixOperations() {
    std::cout << "\n=== Matrix Operations Benchmark ===" << std::endl;
    
    std::vector<std::pair<size_t, size_t>> testSizes = {
        {64, 64},
        {128, 128}, 
        {256, 256},
        {512, 512},
        {1024, 1024}
    };
    
    for (auto [rows, cols] : testSizes) {
        std::cout << "\nMatrix size: " << rows << "x" << cols << std::endl;
        std::cout << "Memory per matrix: " << (rows * cols * sizeof(float) / 1024.0 / 1024.0) << " MB" << std::endl;
        
        // Create test matrices
        Matrix a = Matrix::random(rows, cols, 0.1f);
        Matrix b = Matrix::random(cols, rows, 0.1f);
        
        BenchmarkUtils::Timer timer;
        
        // CPU Matrix Multiplication
        Matrix::setExecutionMode(ExecutionMode::CPU);
        timer.start();
        Matrix result_cpu = a * b;
        double cpu_time = timer.stop();
        
        // GPU Matrix Multiplication (if available)
        double gpu_time = -1;
        Matrix result_gpu(0, 0);
        
        SimpleMetalGPU& gpu = SimpleMetalGPU::getInstance();
        if (MetalUtils::isMetalSupported() && gpu.initialize()) {
            Matrix::setExecutionMode(ExecutionMode::GPU);
            timer.start();
            result_gpu = a * b;
            gpu_time = timer.stop();
            
            // Verify results match (within tolerance)
            bool results_match = true;
            float max_diff = 0.0f;
            for (size_t i = 0; i < std::min(result_cpu.rows, size_t(10)); ++i) {
                for (size_t j = 0; j < std::min(result_cpu.cols, size_t(10)); ++j) {
                    float diff = std::abs(result_cpu[i][j] - result_gpu[i][j]);
                    max_diff = std::max(max_diff, diff);
                    if (diff > 1e-4f) {
                        results_match = false;
                    }
                }
            }
            
            std::cout << "Results match: " << (results_match ? "Yes" : "No");
            if (max_diff > 0) std::cout << " (max diff: " << max_diff << ")";
            std::cout << std::endl;
        }
        
        // Display results
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
        
        if (gpu_time > 0) {
            std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
            std::cout << "Speedup: " << (cpu_time / gpu_time) << "x" << std::endl;
            
            // Calculate throughput
            double cpu_gflops = (2.0 * rows * cols * rows) / (cpu_time * 1e6);  // GEMM is 2*M*N*K FLOPs
            double gpu_gflops = (2.0 * rows * cols * rows) / (gpu_time * 1e6);
            std::cout << "CPU: " << cpu_gflops << " GFLOPS" << std::endl;
            std::cout << "GPU: " << gpu_gflops << " GFLOPS" << std::endl;
        } else {
            std::cout << "GPU not available or failed to initialize" << std::endl;
        }
        
        std::cout << std::string(50, '-') << std::endl;
    }
}

void printSystemInfo() {
    std::cout << "=== System Information ===" << std::endl;
    
    // Metal GPU info
    if (MetalUtils::isMetalSupported()) {
        std::cout << "Metal GPU: Available" << std::endl;
        SimpleMetalGPU& gpu = SimpleMetalGPU::getInstance();
        if (gpu.initialize()) {
            gpu.printPerformanceInfo();
        }
    } else {
        std::cout << "Metal GPU: Not available" << std::endl;
    }
    
    std::cout << "Current execution mode: ";
    switch (Matrix::getExecutionMode()) {
        case ExecutionMode::CPU: std::cout << "CPU" << std::endl; break;
        case ExecutionMode::GPU: std::cout << "GPU" << std::endl; break;
        case ExecutionMode::AUTO: std::cout << "AUTO" << std::endl; break;
    }
    
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== GPU Performance Benchmark ===" << std::endl;
    
    bool run_matrix_bench = true;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --help, -h         Show this help message" << std::endl;
            return 0;
        }
    }
    
    printSystemInfo();
    
    if (run_matrix_bench) {
        benchmarkMatrixOperations();
    }
    
    std::cout << "\n=== Benchmark Complete ===" << std::endl;
    return 0;
}