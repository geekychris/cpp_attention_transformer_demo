#pragma once

#include "transformer.h"
#include <fstream>
#include <chrono>
#include <iomanip>

// Forward declarations
class DataLoader;
class Optimizer;

// Loss functions
namespace Loss {
    float cross_entropy(const Matrix& logits, const std::vector<int>& targets);
    Matrix cross_entropy_grad(const Matrix& logits, const std::vector<int>& targets);
}

// Simple gradient holder for backpropagation
struct Gradients {
    // Matrix gradients
    std::unordered_map<std::string, Matrix> matrix_grads;
    
    void zero();
    void clip(float max_norm = 1.0f);
};

// Simple SGD optimizer
class SGDOptimizer {
private:
    float learning_rate;
    float weight_decay;
    
public:
    SGDOptimizer(float lr = 0.001f, float decay = 0.0f);
    void update(Matrix& weights, const Matrix& gradients);
    void set_learning_rate(float lr) { learning_rate = lr; }
    float get_learning_rate() const { return learning_rate; }
};

// Data loading utilities
class DataLoader {
private:
    std::vector<std::string> data;
    size_t batch_size;
    size_t max_seq_len;
    size_t current_idx;
    SimpleTokenizer* tokenizer;
    
public:
    DataLoader(const std::string& filename, size_t batch_sz, size_t max_len, SimpleTokenizer& tok);
    
    struct Batch {
        std::vector<std::vector<int>> inputs;   // [batch_size, seq_len-1]
        std::vector<std::vector<int>> targets;  // [batch_size, seq_len-1]
        size_t size() const { return inputs.size(); }
    };
    
    bool has_next() const;
    Batch next_batch();
    void reset();
    size_t size() const { return data.size(); }
};

// Training utilities
class TrainingTransformer : public Transformer {
private:
    std::unique_ptr<SGDOptimizer> optimizer;
    Gradients gradients;
    
public:
    TrainingTransformer(size_t vocab_sz, size_t model_dim, size_t heads, size_t layers, size_t ff_dim, size_t max_len);
    
    // Training methods
    float train_step(const DataLoader::Batch& batch);
    float validate(DataLoader& val_loader);
    void train_epoch(DataLoader& train_loader, DataLoader& val_loader, int epoch);
    
    // Simplified backward pass (numerical gradients for demonstration)
    void backward(const Matrix& loss_grad);
    
    // Model serialization
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);
    
    // Setters
    void set_learning_rate(float lr);
    float get_learning_rate() const;
};

// Training configuration
struct TrainingConfig {
    int epochs = 10;
    size_t batch_size = 32;
    float learning_rate = 0.001f;
    float weight_decay = 0.0f;
    int print_every = 100;
    int save_every = 1000;
    std::string model_save_path = "model.bin";
    std::string train_data_path = "train_data.txt";
    std::string val_data_path = "val_data.txt";
};

// Training loop
void train_model(TrainingConfig config);