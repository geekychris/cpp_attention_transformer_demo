#include "transformer.h"

// Main Transformer implementation
Transformer::Transformer(size_t vocab_sz, size_t model_dim, size_t heads, size_t layers, size_t ff_dim, size_t max_len)
    : output_projection(model_dim, vocab_sz), vocab_size(vocab_sz), d_model(model_dim), 
      num_heads(heads), num_layers(layers), d_ff(ff_dim), max_seq_len(max_len) {
    
    // Initialize components
    embedding = std::make_unique<Embedding>(vocab_sz, model_dim);
    pos_encoding = std::make_unique<PositionalEncoding>(max_len, model_dim);
    
    // Create decoder layers (for a GPT-style decoder-only model)
    for (size_t i = 0; i < num_layers; ++i) {
        decoder_layers.push_back(std::make_unique<TransformerDecoderLayer>(model_dim, heads, ff_dim));
    }
    
    initialize_weights();
}

void Transformer::initialize_weights() {
    // Initialize output projection layer
    float std = std::sqrt(1.0f / d_model);
    output_projection.randomize(std);
}

Matrix Transformer::forward(const std::vector<int>& tokens) const {
    size_t seq_len = tokens.size();
    if (seq_len > max_seq_len) {
        throw std::invalid_argument("Sequence length exceeds maximum allowed length");
    }
    
    // Token embeddings
    Matrix token_embeddings = embedding->forward(tokens);  // [seq_len, d_model]
    
    // Positional encodings
    Matrix pos_embeddings = pos_encoding->get_encoding(seq_len);  // [seq_len, d_model]
    
    // Add token and positional embeddings
    Matrix x = token_embeddings + pos_embeddings;  // [seq_len, d_model]
    
    // Pass through decoder layers
    Matrix empty_encoder_output(0, 0);  // For decoder-only model
    for (const auto& layer : decoder_layers) {
        x = layer->forward(x, empty_encoder_output);
    }
    
    // Final output projection to vocabulary
    Matrix logits = x * output_projection;  // [seq_len, vocab_size]
    
    return logits;
}

std::string Transformer::generate(const std::string& prompt, SimpleTokenizer& tokenizer, size_t max_tokens) const {
    // Encode the prompt
    std::vector<int> tokens = tokenizer.encode(prompt);
    
    std::cout << "Generating text with prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "Initial tokens: ";
    for (int token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    
    // Generate tokens one by one
    for (size_t i = 0; i < max_tokens; ++i) {
        if (tokens.size() >= max_seq_len) {
            break;  // Stop if we reach max sequence length
        }
        
        // Forward pass
        Matrix logits = forward(tokens);
        
        // Get logits for the last token position
        size_t last_pos = logits.rows - 1;
        
        // Simple greedy decoding: pick the token with highest probability
        float max_logit = logits[last_pos][0];
        int next_token = 0;
        
        for (size_t j = 1; j < logits.cols; ++j) {
            if (logits[last_pos][j] > max_logit) {
                max_logit = logits[last_pos][j];
                next_token = static_cast<int>(j);
            }
        }
        
        // Stop if we generate EOS token
        if (next_token == SimpleTokenizer::EOS_TOKEN) {
            break;
        }
        
        // Add the new token
        tokens.push_back(next_token);
    }
    
    // Decode the generated tokens
    return tokenizer.decode(tokens);
}

// Helper function to create a small transformer for demonstration
std::unique_ptr<Transformer> create_demo_transformer(SimpleTokenizer& tokenizer) {
    // Small transformer configuration
    size_t vocab_size = std::max(1000, tokenizer.vocab_size());  // Ensure minimum vocab size
    size_t d_model = 128;      // Small model dimension
    size_t num_heads = 4;      // 4 attention heads
    size_t num_layers = 2;     // 2 transformer layers
    size_t d_ff = 512;         // Feed-forward dimension
    size_t max_seq_len = 128;  // Maximum sequence length
    
    return std::make_unique<Transformer>(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len);
}