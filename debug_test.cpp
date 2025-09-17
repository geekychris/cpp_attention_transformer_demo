#include "training.h"
#include <iostream>

int main() {
    std::cout << "Starting debug test..." << std::endl;
    
    try {
        std::cout << "Creating tokenizer..." << std::endl;
        SimpleTokenizer tokenizer;
        std::cout << "Tokenizer created successfully" << std::endl;
        
        std::cout << "Adding test words to tokenizer..." << std::endl;
        for (const auto& word : {"the", "and", "to", "of", "a", "in", "is", "it", "you", "that"}) {
            tokenizer.add_word(word);
        }
        std::cout << "Tokenizer vocab size: " << tokenizer.vocab_size() << std::endl;
        
        std::cout << "Creating model..." << std::endl;
        size_t vocab_size = std::max(size_t(1000), static_cast<size_t>(tokenizer.vocab_size()));
        TrainingTransformer model(vocab_size, 128, 4, 2, 512, 64);
        std::cout << "Model created successfully" << std::endl;
        
        std::cout << "Testing tokenizer encode..." << std::endl;
        std::vector<int> tokens = tokenizer.encode("the quick brown fox");
        std::cout << "Encoded " << tokens.size() << " tokens" << std::endl;
        
        std::cout << "Testing model forward pass..." << std::endl;
        if (!tokens.empty()) {
            Matrix logits = model.forward(tokens);
            std::cout << "Forward pass completed, logits shape: " << logits.rows << "x" << logits.cols << std::endl;
        }
        
        std::cout << "All tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}