#include "training.h"
#include <iostream>
#include <fstream>

int main() {
    std::cout << "=== Inference Analysis ===\n" << std::endl;
    
    try {
        // Initialize tokenizer the same way as training
        SimpleTokenizer tokenizer;
        
        // Pre-populate tokenizer with words that might be in training data
        // This simulates what happens during training when the tokenizer sees new words
        std::ifstream train_file("train_data.txt");
        std::string line;
        int lines_processed = 0;
        
        std::cout << "Building vocabulary from training data..." << std::endl;
        while (std::getline(train_file, line) && lines_processed < 1000) {
            // This will add words to the tokenizer vocabulary
            tokenizer.encode(line);
            lines_processed++;
        }
        
        std::cout << "Vocabulary size after processing training data: " << tokenizer.vocab_size() << std::endl;
        
        // Create model with same architecture as training
        size_t vocab_size = std::max(size_t(1000), static_cast<size_t>(tokenizer.vocab_size()));
        std::cout << "Model vocab size: " << vocab_size << std::endl;
        
        TrainingTransformer model(vocab_size, 128, 4, 2, 512, 64);
        
        // Try to load the trained model
        std::cout << "Loading model..." << std::endl;
        model.load_model("model.bin");
        std::cout << "Model loaded!" << std::endl;
        
        // Test with different prompts
        std::vector<std::string> test_prompts = {
            "the cat",
            "hello world", 
            "once upon a time",
            "the quick brown fox"
        };
        
        for (const auto& prompt : test_prompts) {
            std::cout << "\n--- Testing prompt: \"" << prompt << "\" ---" << std::endl;
            
            // Show token encoding
            std::vector<int> tokens = tokenizer.encode(prompt);
            std::cout << "Tokens: ";
            for (int token : tokens) {
                std::cout << token << " ";
            }
            std::cout << std::endl;
            
            // Generate text
            std::string generated = model.generate(prompt, tokenizer, 10);
            std::cout << "Input:     \"" << prompt << "\"" << std::endl;
            std::cout << "Generated: \"" << generated << "\"" << std::endl;
            
            // Show if anything new was generated
            if (generated.length() > prompt.length()) {
                std::string new_part = generated.substr(prompt.length());
                std::cout << "New part:  \"" << new_part << "\"" << std::endl;
            } else {
                std::cout << "No new text generated beyond the input." << std::endl;
            }
        }
        
        std::cout << "\n=== Analysis ===" << std::endl;
        std::cout << "Why the generation looks similar to the input:" << std::endl;
        std::cout << "1. Small vocabulary - many words become <UNK> tokens" << std::endl;
        std::cout << "2. Undertrained model - hasn't learned meaningful patterns" << std::endl;
        std::cout << "3. Simple greedy decoding - always picks most likely token" << std::endl;
        std::cout << "4. Short training - model needs thousands of steps to learn" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}