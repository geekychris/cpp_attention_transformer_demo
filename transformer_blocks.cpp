#include "transformer.h"

// Feed-forward network implementation
FeedForward::FeedForward(size_t model_dim, size_t ff_dim) 
    : W1(model_dim, ff_dim), b1(1, ff_dim), W2(ff_dim, model_dim), b2(1, model_dim), 
      d_model(model_dim), d_ff(ff_dim) {
    initialize_weights();
}

void FeedForward::initialize_weights() {
    // Xavier initialization
    float std1 = std::sqrt(2.0f / d_model);
    float std2 = std::sqrt(2.0f / d_ff);
    
    W1.randomize(std1);
    W2.randomize(std2);
    b1.zero();
    b2.zero();
}

Matrix FeedForward::forward(const Matrix& x) const {
    // First linear transformation + GELU activation
    Matrix h1 = x * W1;  // [seq_len, d_ff]
    
    // Add bias
    for (size_t i = 0; i < h1.rows; ++i) {
        for (size_t j = 0; j < h1.cols; ++j) {
            h1[i][j] += b1[0][j];
        }
    }
    
    // Apply GELU activation (GPU-accelerated if available)
    h1 = Activations::gelu_matrix(h1);
    
    // Second linear transformation
    Matrix output = h1 * W2;  // [seq_len, d_model]
    
    // Add bias
    for (size_t i = 0; i < output.rows; ++i) {
        for (size_t j = 0; j < output.cols; ++j) {
            output[i][j] += b2[0][j];
        }
    }
    
    return output;
}

// Transformer encoder layer implementation
TransformerEncoderLayer::TransformerEncoderLayer(size_t model_dim, size_t num_heads, size_t ff_dim) 
    : gamma1(1, model_dim), beta1(1, model_dim), gamma2(1, model_dim), beta2(1, model_dim), 
      d_model(model_dim) {
    
    self_attention = std::make_unique<MultiHeadAttention>(model_dim, num_heads);
    feed_forward = std::make_unique<FeedForward>(model_dim, ff_dim);
    initialize_weights();
}

void TransformerEncoderLayer::initialize_weights() {
    // Initialize layer norm parameters
    for (size_t i = 0; i < d_model; ++i) {
        gamma1[0][i] = 1.0f;
        beta1[0][i] = 0.0f;
        gamma2[0][i] = 1.0f;
        beta2[0][i] = 0.0f;
    }
}

Matrix TransformerEncoderLayer::forward(const Matrix& x) const {
    // Self-attention with residual connection and layer norm
    Matrix attn_output = self_attention->forward(x, x, x, false);  // No masking for encoder
    Matrix x1 = x + attn_output;  // Residual connection
    Matrix x1_norm = Activations::layer_norm(x1, gamma1, beta1);  // Layer normalization
    
    // Feed-forward with residual connection and layer norm
    Matrix ff_output = feed_forward->forward(x1_norm);
    Matrix x2 = x1_norm + ff_output;  // Residual connection
    Matrix x2_norm = Activations::layer_norm(x2, gamma2, beta2);  // Layer normalization
    
    return x2_norm;
}

// Transformer decoder layer implementation
TransformerDecoderLayer::TransformerDecoderLayer(size_t model_dim, size_t num_heads, size_t ff_dim) 
    : gamma1(1, model_dim), beta1(1, model_dim), gamma2(1, model_dim), beta2(1, model_dim), 
      gamma3(1, model_dim), beta3(1, model_dim), d_model(model_dim) {
    
    self_attention = std::make_unique<MultiHeadAttention>(model_dim, num_heads);
    cross_attention = std::make_unique<MultiHeadAttention>(model_dim, num_heads);
    feed_forward = std::make_unique<FeedForward>(model_dim, ff_dim);
    initialize_weights();
}

void TransformerDecoderLayer::initialize_weights() {
    // Initialize layer norm parameters
    for (size_t i = 0; i < d_model; ++i) {
        gamma1[0][i] = 1.0f;
        beta1[0][i] = 0.0f;
        gamma2[0][i] = 1.0f;
        beta2[0][i] = 0.0f;
        gamma3[0][i] = 1.0f;
        beta3[0][i] = 0.0f;
    }
}

Matrix TransformerDecoderLayer::forward(const Matrix& x, const Matrix& encoder_output) const {
    // Masked self-attention with residual connection and layer norm
    Matrix self_attn_output = self_attention->forward(x, x, x, true);  // With causal masking
    Matrix x1 = x + self_attn_output;  // Residual connection
    Matrix x1_norm = Activations::layer_norm(x1, gamma1, beta1);  // Layer normalization
    
    // Cross-attention with encoder output (if provided)
    Matrix x2_norm = x1_norm;  // Initialize with correct dimensions
    if (encoder_output.rows > 0) {
        Matrix cross_attn_output = cross_attention->forward(x1_norm, encoder_output, encoder_output, false);
        Matrix x2 = x1_norm + cross_attn_output;  // Residual connection
        x2_norm = Activations::layer_norm(x2, gamma2, beta2);  // Layer normalization
    } else {
        // For decoder-only models (like GPT), skip cross-attention
        x2_norm = x1_norm;
    }
    
    // Feed-forward with residual connection and layer norm
    Matrix ff_output = feed_forward->forward(x2_norm);
    Matrix x3 = x2_norm + ff_output;  // Residual connection
    Matrix x3_norm = Activations::layer_norm(x3, gamma3, beta3);  // Layer normalization
    
    return x3_norm;
}