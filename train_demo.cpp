#include "training.h"
#include <iostream>

int main() {
    std::cout << "=== Simple Training Demo ===" << std::endl;
    std::cout << "This is a simplified training demonstration." << std::endl;
    std::cout << std::endl;
    
    try {
        // Initialize tokenizer
        SimpleTokenizer tokenizer;
        
        // Add some basic vocabulary from our training data
        std::ifstream train_file("train_data.txt");
        std::string line;
        int lines_processed = 0;
        
        while (std::getline(train_file, line) && lines_processed < 100) {
            // Tokenize the line to build vocabulary
            tokenizer.encode(line);
            lines_processed++;
        }
        train_file.close();
        
        std::cout << "Built vocabulary with " << tokenizer.vocab_size() << " tokens" << std::endl;
        
        // Create a small model for demonstration
        size_t vocab_size = std::max(100, tokenizer.vocab_size());
        TrainingTransformer model(vocab_size, 64, 2, 1, 256, 32);
        model.set_learning_rate(0.01f);
        
        std::cout << "Created small model:" << std::endl;
        std::cout << "  - Vocab size: " << vocab_size << std::endl;
        std::cout << "  - Model dim: 64" << std::endl;
        std::cout << "  - Heads: 2" << std::endl;
        std::cout << "  - Layers: 1" << std::endl;
        
        // Test basic forward pass
        std::vector<int> test_tokens = tokenizer.encode("hello world");
        if (test_tokens.size() > 1) {
            std::cout << "\nTesting forward pass..." << std::endl;
            Matrix logits = model.forward(test_tokens);
            std::cout << "Forward pass successful! Output shape: [" 
                      << logits.rows << ", " << logits.cols << "]" << std::endl;
            
            // Test loss calculation
            std::vector<int> target_tokens(test_tokens.begin() + 1, test_tokens.end());
            float loss = Loss::cross_entropy(logits, target_tokens);
            std::cout << "Loss calculation successful! Loss: " << loss << std::endl;
        }
        
        // Save the model
        model.save_model("demo_model.bin");
        
        // Test text generation
        std::cout << "\nTesting text generation..." << std::endl;
        std::string generated = model.generate("hello", tokenizer, 5);
        std::cout << "Generated: \"" << generated << "\"" << std::endl;
        
        std::cout << "\nDemo completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}