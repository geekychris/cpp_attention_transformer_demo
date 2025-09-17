#include "training.h"
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    std::cout << "=== Transformer Training Script ===" << std::endl;
    std::cout << "Starting training process..." << std::endl;
    std::cout << std::endl;

    TrainingConfig config;
    bool useGPU = false;
    bool forceGPU = false;
    bool autoGPU = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--epochs" && i + 1 < argc) {
            config.epochs = std::atoi(argv[++i]);
        } else if (arg == "--batch_size" && i + 1 < argc) {
            config.batch_size = std::atoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            config.learning_rate = std::atof(argv[++i]);
        } else if (arg == "--model" && i + 1 < argc) {
            config.model_save_path = argv[++i];
        } else if (arg == "--train_data" && i + 1 < argc) {
            config.train_data_path = argv[++i];
        } else if (arg == "--val_data" && i + 1 < argc) {
            config.val_data_path = argv[++i];
        } else if (arg == "--gpu") {
            useGPU = true;
            forceGPU = true;
        } else if (arg == "--auto-gpu") {
            useGPU = true;
            autoGPU = true;
        } else if (arg == "--cpu") {
            useGPU = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --epochs N        Number of training epochs (default: 10)" << std::endl;
            std::cout << "  --batch_size N    Batch size (default: 32)" << std::endl;
            std::cout << "  --lr RATE         Learning rate (default: 0.001)" << std::endl;
            std::cout << "  --model PATH      Model save path (default: model.bin)" << std::endl;
            std::cout << "  --train_data PATH Training data path (default: train_data.txt)" << std::endl;
            std::cout << "  --val_data PATH   Validation data path (default: val_data.txt)" << std::endl;
            std::cout << "  --gpu             Force GPU acceleration (Metal on macOS)" << std::endl;
            std::cout << "  --auto-gpu        Use GPU for large matrices, CPU for small ones" << std::endl;
            std::cout << "  --cpu             Force CPU-only computation (default)" << std::endl;
            std::cout << "  --help, -h        Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Configure GPU acceleration
    if (useGPU) {
        if (forceGPU) {
            std::cout << "Enabling GPU acceleration (forced)..." << std::endl;
            Matrix::setExecutionMode(ExecutionMode::GPU);
        } else if (autoGPU) {
            std::cout << "Enabling automatic GPU/CPU selection..." << std::endl;
            Matrix::setExecutionMode(ExecutionMode::AUTO);
        }
    } else {
        std::cout << "Using CPU-only computation..." << std::endl;
        Matrix::setExecutionMode(ExecutionMode::CPU);
    }
    
    try {
        train_model(config);
    } catch (const std::exception& e) {
        std::cerr << "Training error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}