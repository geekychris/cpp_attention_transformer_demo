#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <sstream>

// Forward declaration of MetalGPU
class MetalGPU;

// Global execution mode
enum class ExecutionMode {
    CPU,
    GPU,
    AUTO  // Automatically choose based on matrix size
};

// Simple matrix class for our neural network operations
class Matrix {
public:
    std::vector<std::vector<float>> data;
    size_t rows, cols;
    
    Matrix(size_t r, size_t c) : data(r, std::vector<float>(c, 0.0f)), rows(r), cols(c) {}
    
    Matrix(const std::vector<std::vector<float>>& d) : data(d), rows(d.size()), cols(d.empty() ? 0 : d[0].size()) {}
    
    // Copy constructor
    Matrix(const Matrix& other) : data(other.data), rows(other.rows), cols(other.cols) {}
    
    // Assignment operator
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            data = other.data;
            rows = other.rows;
            cols = other.cols;
        }
        return *this;
    }
    
    // Access operators
    std::vector<float>& operator[](size_t i) { return data[i]; }
    const std::vector<float>& operator[](size_t i) const { return data[i]; }
    
    // Matrix operations
    Matrix operator*(const Matrix& other) const;
    Matrix operator+(const Matrix& other) const;
    Matrix transpose() const;
    void zero();
    void randomize(float std = 0.1f);
    void print() const;
    
    // GPU-accelerated operations
    Matrix multiply_gpu(const Matrix& other) const;
    Matrix add_gpu(const Matrix& other) const;
    Matrix transpose_gpu() const;
    void randomize_gpu(float std = 0.1f);
    
    // Execution mode control
    static void setExecutionMode(ExecutionMode mode);
    static ExecutionMode getExecutionMode();
    static bool shouldUseGPU(size_t rows, size_t cols);
    
    // Static utility functions
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);
    static Matrix random(size_t rows, size_t cols, float std = 0.1f);
    
private:
    static ExecutionMode s_executionMode;
    static const size_t GPU_THRESHOLD = 64; // Minimum matrix size to use GPU
};

// Activation functions
namespace Activations {
    float relu(float x);
    float gelu(float x);
    float gelu_derivative(float x);  // GELU derivative for backprop
    Matrix gelu_matrix(const Matrix& x);  // GPU-accelerated GELU for matrices
    Matrix gelu_derivative_matrix(const Matrix& x);  // GELU derivative for matrices
    Matrix softmax(const Matrix& x);
    Matrix layer_norm(const Matrix& x, const Matrix& gamma, const Matrix& beta, float eps = 1e-5f);
    
    // Layer norm backpropagation
    struct LayerNormGradients {
        Matrix grad_input;
        Matrix grad_gamma;
        Matrix grad_beta;
        
        LayerNormGradients(size_t input_rows, size_t input_cols, size_t param_size) 
            : grad_input(input_rows, input_cols), grad_gamma(param_size, 1), grad_beta(param_size, 1) {}
    };
    LayerNormGradients layer_norm_backward(const Matrix& grad_output, const Matrix& input, 
                                          const Matrix& gamma, const Matrix& beta, float eps = 1e-5f);
}

// Positional encoding for sequence position information
class PositionalEncoding {
private:
    Matrix encoding;
    size_t max_seq_len, d_model;
    
public:
    PositionalEncoding(size_t max_len, size_t model_dim);
    Matrix get_encoding(size_t seq_len) const;
};

// Multi-head attention mechanism
class MultiHeadAttention {
private:
    size_t d_model, num_heads, d_k;
    Matrix W_q, W_k, W_v, W_o;  // Weight matrices for Q, K, V and output
    
    // Gradient storage
    mutable Matrix W_q_grad, W_k_grad, W_v_grad, W_o_grad;
    mutable Matrix cached_input, cached_Q, cached_K, cached_V, cached_attn_weights;
    
    Matrix scaled_dot_product_attention(const Matrix& Q, const Matrix& K, const Matrix& V, bool mask = false) const;
    
public:
    MultiHeadAttention(size_t model_dim, size_t heads);
    Matrix forward(const Matrix& query, const Matrix& key, const Matrix& value, bool mask = false) const;
    void initialize_weights();
    
    // Backpropagation methods
    Matrix backward(const Matrix& grad_output, bool mask = false) const;
    void apply_gradients(float learning_rate);
    void zero_gradients();
};

// Feed-forward network
class FeedForward {
private:
    Matrix W1, b1, W2, b2;
    size_t d_model, d_ff;
    
    // Gradient storage
    mutable Matrix W1_grad, b1_grad, W2_grad, b2_grad;
    mutable Matrix cached_input, cached_h1_pre_gelu, cached_h1_post_gelu;
    
public:
    FeedForward(size_t model_dim, size_t ff_dim);
    Matrix forward(const Matrix& x) const;
    void initialize_weights();
    
    // Backpropagation methods
    Matrix backward(const Matrix& grad_output) const;
    void apply_gradients(float learning_rate);
    void zero_gradients();
};

// Transformer encoder layer
class TransformerEncoderLayer {
private:
    std::unique_ptr<MultiHeadAttention> self_attention;
    std::unique_ptr<FeedForward> feed_forward;
    Matrix gamma1, beta1, gamma2, beta2;  // Layer norm parameters
    size_t d_model;
    
public:
    TransformerEncoderLayer(size_t model_dim, size_t num_heads, size_t ff_dim);
    Matrix forward(const Matrix& x) const;
    void initialize_weights();
};

// Transformer decoder layer (for generative models)
class TransformerDecoderLayer {
private:
    std::unique_ptr<MultiHeadAttention> self_attention;
    std::unique_ptr<MultiHeadAttention> cross_attention;
    std::unique_ptr<FeedForward> feed_forward;
    Matrix gamma1, beta1, gamma2, beta2, gamma3, beta3;  // Layer norm parameters
    size_t d_model;
    
    // Gradient storage for layer norm parameters
    mutable Matrix gamma1_grad, beta1_grad, gamma2_grad, beta2_grad, gamma3_grad, beta3_grad;
    
public:
    TransformerDecoderLayer(size_t model_dim, size_t num_heads, size_t ff_dim);
    Matrix forward(const Matrix& x, const Matrix& encoder_output) const;
    void initialize_weights();
    
    // Backpropagation methods
    Matrix backward(const Matrix& grad_output, const Matrix& encoder_output) const;
    void apply_gradients(float learning_rate);
    void zero_gradients();
};

// Simple tokenizer
class SimpleTokenizer {
private:
    std::unordered_map<std::string, int> word_to_id;
    std::unordered_map<int, std::string> id_to_word;
    int next_id;
    
public:
    static const int PAD_TOKEN = 0;
    static const int UNK_TOKEN = 1;
    static const int BOS_TOKEN = 2;
    static const int EOS_TOKEN = 3;
    
    SimpleTokenizer();
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& tokens);
    int vocab_size() const { return next_id; }
    void add_word(const std::string& word);
    void save_vocab(const std::string& filename) const;
    void load_vocab(const std::string& filename);
};

// Embedding layer
class Embedding {
private:
    Matrix embeddings;
    size_t vocab_size, d_model;
    
public:
    Embedding(size_t vocab_sz, size_t model_dim);
    Matrix forward(const std::vector<int>& tokens) const;
    void initialize_weights();
    
    // Public accessor for embeddings (for training)
    Matrix& get_embeddings() { return embeddings; }
};

// Main Transformer model
class Transformer {
protected:
    std::unique_ptr<Embedding> embedding;
    std::unique_ptr<PositionalEncoding> pos_encoding;
    std::vector<std::unique_ptr<TransformerEncoderLayer>> encoder_layers;
    std::vector<std::unique_ptr<TransformerDecoderLayer>> decoder_layers;
    Matrix output_projection;  // Final linear layer to vocabulary
    
    size_t vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len;
    
public:
    Transformer(size_t vocab_sz, size_t model_dim, size_t heads, size_t layers, size_t ff_dim, size_t max_len);
    
    // Forward pass for language modeling (decoder-only)
    virtual Matrix forward(const std::vector<int>& tokens) const;
    
    // Generate text (simple greedy decoding)
    std::string generate(const std::string& prompt, SimpleTokenizer& tokenizer, size_t max_tokens = 50) const;
    
    void initialize_weights();
};