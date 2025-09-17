#include "training.h"
#include <iostream>
#include <fstream>
#include <cstdlib>

void interactive_mode(TrainingTransformer& model, SimpleTokenizer& tokenizer) {
    std::cout << "\n=== Interactive Mode ===" << std::endl;
    
    // Show current execution mode
    ExecutionMode mode = Matrix::getExecutionMode();
    std::cout << "Execution mode: ";
    switch (mode) {
        case ExecutionMode::CPU: std::cout << "CPU only"; break;
        case ExecutionMode::GPU: std::cout << "GPU accelerated"; break;
        case ExecutionMode::AUTO: std::cout << "Auto (GPU for large matrices)"; break;
    }
    std::cout << std::endl;
    
    std::cout << "Enter text prompts (or 'quit' to exit):" << std::endl;
    
    std::string line;
    while (true) {
        std::cout << "\n> ";
        if (!std::getline(std::cin, line)) {
            break;
        }
        
        if (line == "quit" || line == "exit" || line == "q") {
            break;
        }
        
        if (line.empty()) {
            continue;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        std::string generated = model.generate(line, tokenizer, 20);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Generated: \"" << generated << "\"" << std::endl;
        std::cout << "Time: " << duration.count() << "ms" << std::endl;
    }
}

void batch_mode(TrainingTransformer& model, SimpleTokenizer& tokenizer, const std::string& prompts_file) {
    std::cout << "\n=== Batch Mode ===" << std::endl;
    std::cout << "Reading prompts from: " << prompts_file << std::endl;
    
    // Show current execution mode
    ExecutionMode mode = Matrix::getExecutionMode();
    std::cout << "Execution mode: ";
    switch (mode) {
        case ExecutionMode::CPU: std::cout << "CPU only"; break;
        case ExecutionMode::GPU: std::cout << "GPU accelerated"; break;
        case ExecutionMode::AUTO: std::cout << "Auto (GPU for large matrices)"; break;
    }
    std::cout << std::endl;
    
    std::ifstream file(prompts_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open prompts file: " << prompts_file << std::endl;
        return;
    }
    
    std::string line;
    int prompt_num = 1;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::cout << "\nPrompt " << prompt_num++ << ": \"" << line << "\"" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        std::string generated = model.generate(line, tokenizer, 20);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Generated: \"" << generated << "\"" << std::endl;
        std::cout << "Time: " << duration.count() << "ms" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== Transformer Inference Script ===" << std::endl;
    
    std::string model_path = "model.bin";
    std::string prompts_file = "";
    bool interactive = false;
    bool show_help = false;
    bool useGPU = false;
    bool forceGPU = false;
    bool autoGPU = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--prompts" && i + 1 < argc) {
            prompts_file = argv[++i];
        } else if (arg == "--interactive" || arg == "-i") {
            interactive = true;
        } else if (arg == "--gpu") {
            useGPU = true;
            forceGPU = true;
        } else if (arg == "--auto-gpu") {
            useGPU = true;
            autoGPU = true;
        } else if (arg == "--cpu") {
            useGPU = false;
        } else if (arg == "--help" || arg == "-h") {
            show_help = true;
        }
    }
    
    if (show_help) {
        std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --model PATH       Path to trained model (default: model.bin)" << std::endl;
        std::cout << "  --prompts PATH     File containing prompts for batch generation" << std::endl;
        std::cout << "  --interactive, -i  Run in interactive mode" << std::endl;
        std::cout << "  --gpu              Force GPU acceleration (Metal on macOS)" << std::endl;
        std::cout << "  --auto-gpu         Use GPU for large matrices, CPU for small ones" << std::endl;
        std::cout << "  --cpu              Force CPU-only computation (default)" << std::endl;
        std::cout << "  --help, -h         Show this help message" << std::endl;
        std::cout << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  " << argv[0] << " --model trained_model.bin --prompts test_prompts.txt --gpu" << std::endl;
        std::cout << "  " << argv[0] << " --interactive --auto-gpu" << std::endl;
        std::cout << "  " << argv[0] << " --prompts my_prompts.txt --cpu" << std::endl;
        return 0;
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
        // Initialize tokenizer (should match training)
        SimpleTokenizer tokenizer;
        
        // Create model with same architecture as training
        // Note: In practice, you'd save/load the model architecture too
        size_t vocab_size = std::max(1000, tokenizer.vocab_size());
        TrainingTransformer model(vocab_size, 128, 4, 2, 512, 64);
        
        // Try to load the model
        std::cout << "Loading model from: " << model_path << std::endl;
        model.load_model(model_path);
        
        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << "Vocabulary size: " << tokenizer.vocab_size() << std::endl;
        
        if (interactive) {
            interactive_mode(model, tokenizer);
        } else if (!prompts_file.empty()) {
            batch_mode(model, tokenizer, prompts_file);
        } else {
            // Default: use test_prompts.txt if available
            if (std::ifstream("test_prompts.txt").good()) {
                batch_mode(model, tokenizer, "test_prompts.txt");
            } else {
                std::cout << "No prompts file specified and test_prompts.txt not found." << std::endl;
                std::cout << "Use --interactive for interactive mode or --prompts <file> for batch mode." << std::endl;
                return 1;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\nInference completed!" << std::endl;
    return 0;
}