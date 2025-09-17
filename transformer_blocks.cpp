#include "transformer.h"

// Feed-forward network implementation
FeedForward::FeedForward(size_t model_dim, size_t ff_dim) 
    : W1(model_dim, ff_dim), b1(1, ff_dim), W2(ff_dim, model_dim), b2(1, model_dim), 
      d_model(model_dim), d_ff(ff_dim),
      W1_grad(model_dim, ff_dim), b1_grad(1, ff_dim), W2_grad(ff_dim, model_dim), b2_grad(1, model_dim),
      cached_input(0, 0), cached_h1_pre_gelu(0, 0), cached_h1_post_gelu(0, 0) {
    initialize_weights();
    zero_gradients();
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
    // Cache input for backprop
    cached_input = x;
    
    // First linear transformation
    Matrix h1_pre = x * W1;  // [seq_len, d_ff]
    
    // Add bias
    for (size_t i = 0; i < h1_pre.rows; ++i) {
        for (size_t j = 0; j < h1_pre.cols; ++j) {
            h1_pre[i][j] += b1[0][j];
        }
    }
    
    // Cache pre-GELU activations
    cached_h1_pre_gelu = h1_pre;
    
    // Apply GELU activation (GPU-accelerated if available)
    Matrix h1_post = Activations::gelu_matrix(h1_pre);
    cached_h1_post_gelu = h1_post;
    
    // Second linear transformation
    Matrix output = h1_post * W2;  // [seq_len, d_model]
    
    // Add bias
    for (size_t i = 0; i < output.rows; ++i) {
        for (size_t j = 0; j < output.cols; ++j) {
            output[i][j] += b2[0][j];
        }
    }
    
    return output;
}

Matrix FeedForward::backward(const Matrix& grad_output) const {
    // grad_output is [seq_len, d_model]
    // Need to backprop through: output = h1_post * W2 + b2
    
    // Gradient w.r.t. W2: h1_post^T @ grad_output
    Matrix h1_post_T = cached_h1_post_gelu.transpose();  // [d_ff, seq_len]
    W2_grad = h1_post_T * grad_output;  // [d_ff, d_model]
    
    // Gradient w.r.t. b2: sum over sequence dimension
    b2_grad.zero();
    for (size_t i = 0; i < grad_output.rows; ++i) {
        for (size_t j = 0; j < grad_output.cols; ++j) {
            b2_grad[0][j] += grad_output[i][j];
        }
    }
    
    // Gradient w.r.t. h1_post: grad_output @ W2^T
    Matrix W2_T = W2.transpose();  // [d_model, d_ff]
    Matrix grad_h1_post = grad_output * W2_T;  // [seq_len, d_ff]
    
    // Gradient w.r.t. h1_pre (through GELU): grad_h1_post * gelu'(h1_pre)
    Matrix gelu_grad = Activations::gelu_derivative_matrix(cached_h1_pre_gelu);
    Matrix grad_h1_pre(grad_h1_post.rows, grad_h1_post.cols);
    for (size_t i = 0; i < grad_h1_post.rows; ++i) {
        for (size_t j = 0; j < grad_h1_post.cols; ++j) {
            grad_h1_pre[i][j] = grad_h1_post[i][j] * gelu_grad[i][j];
        }
    }
    
    // Gradient w.r.t. W1: input^T @ grad_h1_pre
    Matrix input_T = cached_input.transpose();  // [d_model, seq_len]
    W1_grad = input_T * grad_h1_pre;  // [d_model, d_ff]
    
    // Gradient w.r.t. b1: sum over sequence dimension
    b1_grad.zero();
    for (size_t i = 0; i < grad_h1_pre.rows; ++i) {
        for (size_t j = 0; j < grad_h1_pre.cols; ++j) {
            b1_grad[0][j] += grad_h1_pre[i][j];
        }
    }
    
    // Gradient w.r.t. input: grad_h1_pre @ W1^T
    Matrix W1_T = W1.transpose();  // [d_ff, d_model]
    Matrix grad_input = grad_h1_pre * W1_T;  // [seq_len, d_model]
    
    return grad_input;
}

void FeedForward::apply_gradients(float learning_rate) {
    // Update weights using accumulated gradients
    for (size_t i = 0; i < W1.rows; ++i) {
        for (size_t j = 0; j < W1.cols; ++j) {
            W1[i][j] -= learning_rate * W1_grad[i][j];
        }
    }
    
    for (size_t i = 0; i < W2.rows; ++i) {
        for (size_t j = 0; j < W2.cols; ++j) {
            W2[i][j] -= learning_rate * W2_grad[i][j];
        }
    }
    
    for (size_t j = 0; j < b1.cols; ++j) {
        b1[0][j] -= learning_rate * b1_grad[0][j];
    }
    
    for (size_t j = 0; j < b2.cols; ++j) {
        b2[0][j] -= learning_rate * b2_grad[0][j];
    }
}

void FeedForward::zero_gradients() {
    W1_grad.zero();
    W2_grad.zero();
    b1_grad.zero();
    b2_grad.zero();
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
      gamma3(1, model_dim), beta3(1, model_dim), d_model(model_dim),
      gamma1_grad(1, model_dim), beta1_grad(1, model_dim), gamma2_grad(1, model_dim), 
      beta2_grad(1, model_dim), gamma3_grad(1, model_dim), beta3_grad(1, model_dim) {
    
    self_attention = std::make_unique<MultiHeadAttention>(model_dim, num_heads);
    cross_attention = std::make_unique<MultiHeadAttention>(model_dim, num_heads);
    feed_forward = std::make_unique<FeedForward>(model_dim, ff_dim);
    initialize_weights();
    zero_gradients();
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

Matrix TransformerDecoderLayer::backward(const Matrix& grad_output, const Matrix& encoder_output) const {
    // For now, implement a simplified version that focuses on feed-forward backprop
    // Full attention backprop is complex and can be added later
    
    // Assume we're in decoder-only mode (encoder_output is empty)
    // Backprop through: LayerNorm -> FeedForward -> LayerNorm -> SelfAttention -> Input
    
    // For simplicity, we'll just backprop through the feed-forward layer for now
    // This still gives us meaningful gradient flow
    
    // Skip the final layer norm backprop for now (complex)
    Matrix grad_ff_input = grad_output;
    
    // Backprop through feed-forward layer
    Matrix grad_ln2_output = feed_forward->backward(grad_ff_input);
    
    // Skip the middle layer norm backprop for now
    Matrix grad_attn_input = grad_ln2_output;
    
    // For now, just pass gradients through (no attention backprop yet)
    // In a complete implementation, this would backprop through self-attention
    Matrix grad_input = grad_attn_input;
    
    return grad_input;
}

void TransformerDecoderLayer::apply_gradients(float learning_rate) {
    // Apply gradients to feed-forward layer
    feed_forward->apply_gradients(learning_rate);
    
    // Apply gradients to layer norm parameters (simplified - no actual gradients computed yet)
    // In a complete implementation, we'd compute and apply layer norm gradients
}

void TransformerDecoderLayer::zero_gradients() {
    // Zero gradients for all components
    feed_forward->zero_gradients();
    
    gamma1_grad.zero();
    beta1_grad.zero();
    gamma2_grad.zero();
    beta2_grad.zero();
    gamma3_grad.zero();
    beta3_grad.zero();
}
