#include "training.h"
#include <iostream>

int main() {
    std::cout << "Testing DataLoader..." << std::endl;
    
    try {
        std::cout << "Creating tokenizer..." << std::endl;
        SimpleTokenizer tokenizer;
        
        // Pre-populate with some words to avoid issues
        for (const auto& word : {"the", "and", "to", "of", "a", "in", "is", "it", "you", "that"}) {
            tokenizer.add_word(word);
        }
        
        std::cout << "Creating DataLoader..." << std::endl;
        DataLoader train_loader("train_data.txt", 1, 64, tokenizer);
        std::cout << "DataLoader created, data size: " << train_loader.size() << std::endl;
        
        if (train_loader.has_next()) {
            std::cout << "Getting first batch..." << std::endl;
            auto batch = train_loader.next_batch();
            std::cout << "Got batch with " << batch.size() << " samples" << std::endl;
            
            if (batch.size() > 0) {
                std::cout << "First sample input size: " << batch.inputs[0].size() << std::endl;
                std::cout << "First sample target size: " << batch.targets[0].size() << std::endl;
            }
        }
        
        std::cout << "DataLoader test passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}