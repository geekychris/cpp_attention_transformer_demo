#include "transformer.h"
#include <iostream>
#include <chrono>

// Forward declaration
std::unique_ptr<Transformer> create_demo_transformer(SimpleTokenizer& tokenizer);

void demonstrate_attention_mechanism() {
    std::cout << "=== Transformer Architecture and Attention Mechanism Demo ===" << std::endl;
    std::cout << std::endl;
    
    std::cout << "This implementation demonstrates key concepts:" << std::endl;
    std::cout << "1. Multi-Head Self-Attention: The core mechanism that allows tokens to 'attend' to each other" << std::endl;
    std::cout << "2. Scaled Dot-Product Attention: QK^T/sqrt(d_k) mechanism" << std::endl;
    std::cout << "3. Positional Encoding: Sinusoidal encodings to represent token positions" << std::endl;
    std::cout << "4. Layer Normalization and Residual Connections: For stable training" << std::endl;
    std::cout << "5. Feed-Forward Networks: For non-linear transformations" << std::endl;
    std::cout << std::endl;
}

void demonstrate_matrix_operations() {
    std::cout << "=== Matrix Operations Demo ===" << std::endl;
    
    // Create some sample matrices
    Matrix A = Matrix::random(3, 4, 0.5f);
    Matrix B = Matrix::random(4, 2, 0.5f);
    
    std::cout << "Matrix A (3x4):" << std::endl;
    A.print();
    
    std::cout << "Matrix B (4x2):" << std::endl;
    B.print();
    
    std::cout << "Matrix multiplication A * B:" << std::endl;
    Matrix C = A * B;
    C.print();
    
    std::cout << "Softmax example:" << std::endl;
    Matrix test_scores(2, 3);
    test_scores[0][0] = 1.0f; test_scores[0][1] = 2.0f; test_scores[0][2] = 3.0f;
    test_scores[1][0] = 4.0f; test_scores[1][1] = 1.0f; test_scores[1][2] = 2.0f;
    
    std::cout << "Before softmax:" << std::endl;
    test_scores.print();
    
    Matrix softmax_result = Activations::softmax(test_scores);
    std::cout << "After softmax:" << std::endl;
    softmax_result.print();
    std::cout << std::endl;
}

void demonstrate_tokenizer() {
    std::cout << "=== Tokenizer Demo ===" << std::endl;
    
    SimpleTokenizer tokenizer;
    
    std::string test_text = "Hello world! This is a simple test of the tokenizer.";
    std::cout << "Original text: \"" << test_text << "\"" << std::endl;
    
    std::vector<int> tokens = tokenizer.encode(test_text);
    std::cout << "Encoded tokens: ";
    for (int token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    
    std::string decoded = tokenizer.decode(tokens);
    std::cout << "Decoded text: \"" << decoded << "\"" << std::endl;
    std::cout << "Vocabulary size: " << tokenizer.vocab_size() << std::endl;
    std::cout << std::endl;
}

void demonstrate_positional_encoding() {
    std::cout << "=== Positional Encoding Demo ===" << std::endl;
    
    PositionalEncoding pos_enc(10, 8);  // Max length 10, model dim 8
    Matrix encoding = pos_enc.get_encoding(5);
    
    std::cout << "Positional encoding for sequence length 5, model dimension 8:" << std::endl;
    encoding.print();
    std::cout << std::endl;
}

void demonstrate_attention() {
    std::cout << "=== Attention Mechanism Demo ===" << std::endl;
    
    // Create a small attention layer
    MultiHeadAttention attention(64, 4);  // 64-dim model, 4 heads
    
    // Create sample input (3 tokens, 64 dimensions each)
    Matrix input = Matrix::random(3, 64, 0.1f);
    std::cout << "Input sequence (3 tokens, 64 dims - showing first few values):" << std::endl;
    input.print();
    
    // Apply attention (self-attention)
    Matrix output = attention.forward(input, input, input, false);
    std::cout << "Attention output (showing first few values):" << std::endl;
    output.print();
    std::cout << std::endl;
}

void demonstrate_text_generation() {
    std::cout << "=== Text Generation Demo ===" << std::endl;
    
    SimpleTokenizer tokenizer;
    
    // Pre-populate tokenizer with some common words
    std::vector<std::string> common_words = {
        "the", "and", "to", "of", "a", "in", "is", "it", "you", "that",
        "he", "was", "for", "on", "are", "as", "with", "his", "they", "i",
        "at", "be", "this", "have", "from", "or", "one", "had", "by", "word",
        "but", "not", "what", "all", "were", "we", "when", "your", "can", "said"
    };
    
    for (const std::string& word : common_words) {
        tokenizer.add_word(word);
    }
    
    std::cout << "Creating small transformer model..." << std::endl;
    auto transformer = create_demo_transformer(tokenizer);
    
    std::cout << "Model created with:" << std::endl;
    std::cout << "- Vocabulary size: " << tokenizer.vocab_size() << std::endl;
    std::cout << "- Model dimension: 128" << std::endl;
    std::cout << "- Number of heads: 4" << std::endl;
    std::cout << "- Number of layers: 2" << std::endl;
    std::cout << "- Feed-forward dimension: 512" << std::endl;
    std::cout << std::endl;
    
    // Test with different prompts
    std::vector<std::string> prompts = {
        "the quick brown",
        "hello world",
        "artificial intelligence"
    };
    
    for (const std::string& prompt : prompts) {
        std::cout << "Generating text with prompt: \"" << prompt << "\"" << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        std::string generated = transformer->generate(prompt, tokenizer, 10);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Generated: \"" << generated << "\"" << std::endl;
        std::cout << "Generation time: " << duration.count() << " ms" << std::endl;
        std::cout << std::endl;
    }
    
    std::cout << "Note: This is an untrained model, so the output is random." << std::endl;
    std::cout << "In a real implementation, you would train the model on a large dataset." << std::endl;
}

int main() {
    std::cout << "C++ Transformer and Attention Implementation" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << std::endl;
    
    try {
        demonstrate_attention_mechanism();
        demonstrate_matrix_operations();
        demonstrate_tokenizer();
        demonstrate_positional_encoding();
        demonstrate_attention();
        demonstrate_text_generation();
        
        std::cout << "Demo completed successfully!" << std::endl;
        std::cout << std::endl;
        std::cout << "Key Learning Points:" << std::endl;
        std::cout << "1. Attention allows each token to look at all other tokens in the sequence" << std::endl;
        std::cout << "2. Multi-head attention captures different types of relationships" << std::endl;
        std::cout << "3. Positional encoding gives the model information about token order" << std::endl;
        std::cout << "4. Layer normalization and residual connections help with training stability" << std::endl;
        std::cout << "5. The transformer architecture is highly parallelizable" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}