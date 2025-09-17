#include "transformer.h"

// Multi-head attention implementation
MultiHeadAttention::MultiHeadAttention(size_t model_dim, size_t heads) 
    : d_model(model_dim), num_heads(heads), W_q(model_dim, model_dim), 
      W_k(model_dim, model_dim), W_v(model_dim, model_dim), W_o(model_dim, model_dim) {
    
    if (model_dim % heads != 0) {
        throw std::invalid_argument("Model dimension must be divisible by number of heads");
    }
    d_k = model_dim / heads;
    initialize_weights();
}

void MultiHeadAttention::initialize_weights() {
    // Initialize with Xavier/Glorot initialization
    float std = std::sqrt(2.0f / d_model);
    W_q.randomize(std);
    W_k.randomize(std);
    W_v.randomize(std);
    W_o.randomize(std);
}

Matrix MultiHeadAttention::scaled_dot_product_attention(const Matrix& Q, const Matrix& K, const Matrix& V, bool mask) const {
    // Q, K, V are of shape [seq_len, d_k]
    size_t seq_len = Q.rows;
    
    // Compute attention scores: QK^T / sqrt(d_k)
    Matrix K_T = K.transpose();  // [d_k, seq_len]
    Matrix scores = Q * K_T;     // [seq_len, seq_len]
    
    // Scale by sqrt(d_k)
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
    for (size_t i = 0; i < scores.rows; ++i) {
        for (size_t j = 0; j < scores.cols; ++j) {
            scores[i][j] *= scale;
        }
    }
    
    // Apply causal mask for decoder (prevents looking at future tokens)
    if (mask) {
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = i + 1; j < seq_len; ++j) {
                scores[i][j] = -1e9f;  // Large negative value
            }
        }
    }
    
    // Apply softmax to get attention weights
    Matrix attention_weights = Activations::softmax(scores);
    
    // Apply attention weights to values
    Matrix output = attention_weights * V;  // [seq_len, d_k]
    
    return output;
}

Matrix MultiHeadAttention::forward(const Matrix& query, const Matrix& key, const Matrix& value, bool mask) const {
    size_t seq_len = query.rows;
    
    // Linear projections
    Matrix Q = query * W_q;  // [seq_len, d_model]
    Matrix K = key * W_k;    // [seq_len, d_model]
    Matrix V = value * W_v;  // [seq_len, d_model]
    
    // Reshape for multi-head attention
    // In a full implementation, we would split into heads and run in parallel
    // For simplicity, we'll process each head sequentially
    
    std::vector<Matrix> head_outputs;
    
    for (size_t h = 0; h < num_heads; ++h) {
        // Extract the portion for this head
        size_t start_col = h * d_k;
        
        Matrix Q_h(seq_len, d_k);
        Matrix K_h(seq_len, d_k);
        Matrix V_h(seq_len, d_k);
        
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < d_k; ++j) {
                Q_h[i][j] = Q[i][start_col + j];
                K_h[i][j] = K[i][start_col + j];
                V_h[i][j] = V[i][start_col + j];
            }
        }
        
        // Apply scaled dot-product attention
        Matrix head_output = scaled_dot_product_attention(Q_h, K_h, V_h, mask);
        head_outputs.push_back(head_output);
    }
    
    // Concatenate heads
    Matrix concat_heads(seq_len, d_model);
    for (size_t h = 0; h < num_heads; ++h) {
        size_t start_col = h * d_k;
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < d_k; ++j) {
                concat_heads[i][start_col + j] = head_outputs[h][i][j];
            }
        }
    }
    
    // Final linear projection
    Matrix output = concat_heads * W_o;
    
    return output;
}