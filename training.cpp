#include "training.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <random>

// Loss functions implementation
namespace Loss {
    float cross_entropy(const Matrix& logits, const std::vector<int>& targets) {
        float total_loss = 0.0f;
        size_t seq_len = std::min(logits.rows, targets.size());
        
        for (size_t i = 0; i < seq_len; ++i) {
            int target = targets[i];
            if (target >= 0 && target < static_cast<int>(logits.cols)) {
                // Apply softmax to get probabilities
                std::vector<float> probs(logits.cols);
                float max_logit = *std::max_element(logits[i].begin(), logits[i].end());
                float sum_exp = 0.0f;
                
                for (size_t j = 0; j < logits.cols; ++j) {
                    probs[j] = std::exp(logits[i][j] - max_logit);
                    sum_exp += probs[j];
                }
                
                for (size_t j = 0; j < logits.cols; ++j) {
                    probs[j] /= sum_exp;
                }
                
                // Cross entropy loss
                total_loss -= std::log(probs[target] + 1e-10f);
            }
        }
        
        return total_loss / seq_len;
    }
    
    Matrix cross_entropy_grad(const Matrix& logits, const std::vector<int>& targets) {
        Matrix grad(logits.rows, logits.cols);
        size_t seq_len = std::min(logits.rows, targets.size());
        
        for (size_t i = 0; i < seq_len; ++i) {
            // Compute softmax
            std::vector<float> probs(logits.cols);
            float max_logit = *std::max_element(logits[i].begin(), logits[i].end());
            float sum_exp = 0.0f;
            
            for (size_t j = 0; j < logits.cols; ++j) {
                probs[j] = std::exp(logits[i][j] - max_logit);
                sum_exp += probs[j];
            }
            
            for (size_t j = 0; j < logits.cols; ++j) {
                probs[j] /= sum_exp;
                grad[i][j] = probs[j];
            }
            
            // Subtract 1 from target class
            int target = targets[i];
            if (target >= 0 && target < static_cast<int>(logits.cols)) {
                grad[i][target] -= 1.0f;
            }
        }
        
        // Scale by sequence length
        for (size_t i = 0; i < grad.rows; ++i) {
            for (size_t j = 0; j < grad.cols; ++j) {
                grad[i][j] /= seq_len;
            }
        }
        
        return grad;
    }
}

// Gradients implementation
void Gradients::zero() {
    for (auto& [name, grad] : matrix_grads) {
        grad.zero();
    }
}

void Gradients::clip(float max_norm) {
    // Simple gradient clipping
    float total_norm = 0.0f;
    
    // Compute total norm
    for (const auto& [name, grad] : matrix_grads) {
        for (size_t i = 0; i < grad.rows; ++i) {
            for (size_t j = 0; j < grad.cols; ++j) {
                total_norm += grad[i][j] * grad[i][j];
            }
        }
    }
    total_norm = std::sqrt(total_norm);
    
    // Clip if necessary
    if (total_norm > max_norm) {
        float clip_factor = max_norm / total_norm;
        for (auto& [name, grad] : matrix_grads) {
            for (size_t i = 0; i < grad.rows; ++i) {
                for (size_t j = 0; j < grad.cols; ++j) {
                    grad[i][j] *= clip_factor;
                }
            }
        }
    }
}

// SGD Optimizer implementation
SGDOptimizer::SGDOptimizer(float lr, float decay, float max_norm) 
    : learning_rate(lr), weight_decay(decay), max_grad_norm(max_norm) {}

void SGDOptimizer::clip_gradients(Matrix& gradients) const {
    // Compute gradient norm
    float grad_norm = 0.0f;
    for (size_t i = 0; i < gradients.rows; ++i) {
        for (size_t j = 0; j < gradients.cols; ++j) {
            grad_norm += gradients[i][j] * gradients[i][j];
        }
    }
    grad_norm = std::sqrt(grad_norm);
    
    // Clip if necessary
    if (grad_norm > max_grad_norm) {
        float clip_factor = max_grad_norm / grad_norm;
        for (size_t i = 0; i < gradients.rows; ++i) {
            for (size_t j = 0; j < gradients.cols; ++j) {
                gradients[i][j] *= clip_factor;
            }
        }
    }
}

void SGDOptimizer::update(Matrix& weights, const Matrix& gradients) {
    // Create a copy for clipping (don't modify the input)
    Matrix clipped_gradients = gradients;
    
    // Apply gradient clipping
    clip_gradients(clipped_gradients);
    
    for (size_t i = 0; i < weights.rows; ++i) {
        for (size_t j = 0; j < weights.cols; ++j) {
            // Apply weight decay
            if (weight_decay > 0.0f) {
                weights[i][j] *= (1.0f - weight_decay * learning_rate);
            }
            
            // Apply gradient update with clipped gradients
            weights[i][j] -= learning_rate * clipped_gradients[i][j];
        }
    }
}

// DataLoader implementation
DataLoader::DataLoader(const std::string& filename, size_t batch_sz, size_t max_len, SimpleTokenizer& tok)
    : batch_size(batch_sz), max_seq_len(max_len), current_idx(0), tokenizer(&tok) {
    
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        if (!line.empty()) {
            data.push_back(line);
        }
    }
    
    std::cout << "Loaded " << data.size() << " lines from " << filename << std::endl;
    
    // CRITICAL FIX: Build vocabulary from all training data during construction
    std::cout << "Building vocabulary from " << data.size() << " lines..." << std::endl;
    for (const auto& text : data) {
        tokenizer->encode(text);  // This builds the vocabulary as a side effect
    }
    std::cout << "Vocabulary built: " << tokenizer->vocab_size() << " tokens" << std::endl;
}

bool DataLoader::has_next() const {
    return current_idx < data.size();
}

DataLoader::Batch DataLoader::next_batch() {
    Batch batch;
    
    size_t end_idx = std::min(current_idx + batch_size, data.size());
    
    for (size_t i = current_idx; i < end_idx; ++i) {
        std::vector<int> tokens = tokenizer->encode(data[i]);
        
        // Truncate or pad to max_seq_len
        if (tokens.size() > max_seq_len) {
            tokens.resize(max_seq_len);
        }
        
        // For language modeling: input is tokens[:-1], target is tokens[1:]
        if (tokens.size() > 1) {
            std::vector<int> input_tokens(tokens.begin(), tokens.end() - 1);
            std::vector<int> target_tokens(tokens.begin() + 1, tokens.end());
            
            batch.inputs.push_back(input_tokens);
            batch.targets.push_back(target_tokens);
        }
    }
    
    current_idx = end_idx;
    return batch;
}

void DataLoader::reset() {
    current_idx = 0;
    
    // Shuffle data for next epoch
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
}

// TrainingTransformer implementation
TrainingTransformer::TrainingTransformer(size_t vocab_sz, size_t model_dim, size_t heads, size_t layers, size_t ff_dim, size_t max_len)
    : Transformer(vocab_sz, model_dim, heads, layers, ff_dim, max_len), 
      cached_transformer_output(0, 0) {
    // Initialize with more stable parameters: lower learning rate, gradient clipping
    optimizer = std::make_unique<SGDOptimizer>(0.0001f, 0.0f, 0.5f);  // lr=0.0001, no weight decay, max_grad_norm=0.5
}

Matrix TrainingTransformer::forward(const std::vector<int>& tokens) const {
    // Cache input tokens for backprop
    cached_input_tokens = tokens;
    
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
    
    // Cache the transformer layer output (before final projection)
    cached_transformer_output = x;
    
    // Final output projection to vocabulary
    Matrix logits = x * output_projection;  // [seq_len, vocab_size]
    
    return logits;
}

float TrainingTransformer::train_step(const DataLoader::Batch& batch) {
    float total_loss = 0.0f;
    
    for (size_t b = 0; b < batch.size(); ++b) {
        const auto& input_tokens = batch.inputs[b];
        const auto& target_tokens = batch.targets[b];
        
        // Forward pass
        Matrix logits = forward(input_tokens);
        
        // Compute loss
        float loss = Loss::cross_entropy(logits, target_tokens);
        
        // Check for NaN or exploding loss
        if (std::isnan(loss) || loss > 50.0f) {
            std::cout << "\n⚠️  Unstable loss detected: " << loss << ". Skipping this batch." << std::endl;
            continue;  // Skip this batch
        }
        
        total_loss += loss;
        
        // Simple numerical gradients for demonstration
        // In practice, you'd implement proper backpropagation
        Matrix loss_grad = Loss::cross_entropy_grad(logits, target_tokens);
        backward(loss_grad);
    }
    
    return total_loss / batch.size();
}

void TrainingTransformer::backward(const Matrix& loss_grad) {
    // NOW IMPLEMENTING PROPER BACKPROPAGATION! 🚀
    // loss_grad is [seq_len, vocab_size] from cross-entropy loss
    // output_projection is [d_model, vocab_size]
    // cached_transformer_output is [seq_len, d_model]
    
    if (cached_transformer_output.rows == 0) {
        std::cerr << "Error: No cached activations for backprop!" << std::endl;
        return;
    }
    
    // Compute gradient for output projection: W_grad = X^T @ loss_grad
    // Where X is transformer output [seq_len, d_model] and loss_grad is [seq_len, vocab_size]
    // Result should be [d_model, vocab_size]
    
    Matrix X_T = cached_transformer_output.transpose();  // [d_model, seq_len]
    Matrix proj_grad = X_T * loss_grad;  // [d_model, vocab_size]
    
    // Normalize by sequence length for stability
    float seq_len = static_cast<float>(loss_grad.rows);
    for (size_t i = 0; i < proj_grad.rows; ++i) {
        for (size_t j = 0; j < proj_grad.cols; ++j) {
            proj_grad[i][j] /= seq_len;
        }
    }
    
    // Apply gradient update to output projection
    optimizer->update(output_projection, proj_grad);
    
    // Compute gradients for transformer layers
    // gradient w.r.t. transformer output: loss_grad @ output_projection^T
    Matrix output_proj_T = output_projection.transpose();  // [vocab_size, d_model]
    Matrix transformer_grad = loss_grad * output_proj_T;  // [seq_len, d_model]
    
    // Backpropagate through decoder layers (in reverse order)
    Matrix current_grad = transformer_grad;
    Matrix empty_encoder_output(0, 0);
    
    for (int i = static_cast<int>(decoder_layers.size()) - 1; i >= 0; --i) {
        current_grad = decoder_layers[i]->backward(current_grad, empty_encoder_output);
        decoder_layers[i]->apply_gradients(optimizer->get_learning_rate());
    }
    
    // Update embedding weights based on input tokens
    // current_grad now contains gradients w.r.t. the transformer input (after embeddings)
    Matrix& embeddings_matrix = embedding->get_embeddings();
    for (size_t t = 0; t < cached_input_tokens.size(); ++t) {
        int token_id = cached_input_tokens[t];
        if (token_id >= 0 && static_cast<size_t>(token_id) < embeddings_matrix.rows && t < current_grad.rows) {
            // Update embedding for this token using gradients from transformer layers
            for (size_t d = 0; d < current_grad.cols && d < embeddings_matrix.cols; ++d) {
                float grad = current_grad[t][d] / static_cast<float>(cached_input_tokens.size());
                embeddings_matrix[token_id][d] -= optimizer->get_learning_rate() * grad;
            }
        }
    }
}

float TrainingTransformer::validate(DataLoader& val_loader) {
    float total_loss = 0.0f;
    int num_batches = 0;
    
    val_loader.reset();
    while (val_loader.has_next()) {
        auto batch = val_loader.next_batch();
        
        float batch_loss = 0.0f;
        for (size_t b = 0; b < batch.size(); ++b) {
            const auto& input_tokens = batch.inputs[b];
            const auto& target_tokens = batch.targets[b];
            
            Matrix logits = forward(input_tokens);
            float loss = Loss::cross_entropy(logits, target_tokens);
            batch_loss += loss;
        }
        
        total_loss += batch_loss / batch.size();
        num_batches++;
        
        // Don't validate on entire dataset for speed
        if (num_batches >= 10) break;
    }
    
    return total_loss / num_batches;
}

void TrainingTransformer::train_epoch(DataLoader& train_loader, DataLoader& val_loader, int epoch) {
    train_loader.reset();
    
    size_t total_batches = train_loader.get_total_batches();
    std::cout << "\n=== Epoch " << epoch << " ===" << std::endl;
    std::cout << "Training on " << total_batches << " batches..." << std::endl;
    
    float epoch_loss = 0.0f;
    int step = 0;
    
    auto epoch_start = std::chrono::high_resolution_clock::now();
    
    while (train_loader.has_next()) {
        auto batch = train_loader.next_batch();
        
        float batch_loss = train_step(batch);
        
        // Check for training instability
        if (std::isnan(batch_loss) || batch_loss > 100.0f) {
            std::cout << "\n❌ Training instability detected. Stopping epoch early." << std::endl;
            std::cout << "   Final loss: " << batch_loss << " at step " << step << std::endl;
            break;
        }
        
        epoch_loss += batch_loss;
        step++;
        
        // Show progress every 20 steps or at key milestones
        if (step % 20 == 0 || step == 1 || static_cast<size_t>(step) == total_batches) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - epoch_start);
            float progress = (float)step / total_batches * 100.0f;
            std::cout << "  [" << std::fixed << std::setprecision(1) << progress << "%] "
                      << "Step " << std::setw(3) << step << "/" << total_batches 
                      << " | Loss: " << std::setprecision(4) << batch_loss
                      << " | Elapsed: " << elapsed.count() << "s" << std::endl;
        }
    }
    
    float val_loss = validate(val_loader);
    
    auto epoch_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);
    
    std::cout << "\n✓ Epoch " << epoch << " Complete" << std::endl;
    std::cout << "  Train Loss: " << std::fixed << std::setprecision(4) << (epoch_loss / step) 
              << " | Val Loss: " << val_loss 
              << " | Duration: " << duration.count() << "s" << std::endl;
}

void TrainingTransformer::save_model(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    
    // Save basic model configuration
    file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
    file.write(reinterpret_cast<const char*>(&d_model), sizeof(d_model));
    file.write(reinterpret_cast<const char*>(&num_heads), sizeof(num_heads));
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    file.write(reinterpret_cast<const char*>(&d_ff), sizeof(d_ff));
    file.write(reinterpret_cast<const char*>(&max_seq_len), sizeof(max_seq_len));
    
    // Save output projection matrix
    file.write(reinterpret_cast<const char*>(&output_projection.rows), sizeof(output_projection.rows));
    file.write(reinterpret_cast<const char*>(&output_projection.cols), sizeof(output_projection.cols));
    
    for (size_t i = 0; i < output_projection.rows; ++i) {
        for (size_t j = 0; j < output_projection.cols; ++j) {
            file.write(reinterpret_cast<const char*>(&output_projection[i][j]), sizeof(float));
        }
    }
    
    file.close();
    std::cout << "Model saved to " << filename << std::endl;
}

void TrainingTransformer::load_model(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open model file: " << filename << std::endl;
        return;
    }
    
    // Load model configuration
    size_t loaded_vocab_size, loaded_d_model, loaded_num_heads, loaded_num_layers, loaded_d_ff, loaded_max_seq_len;
    
    file.read(reinterpret_cast<char*>(&loaded_vocab_size), sizeof(loaded_vocab_size));
    file.read(reinterpret_cast<char*>(&loaded_d_model), sizeof(loaded_d_model));
    file.read(reinterpret_cast<char*>(&loaded_num_heads), sizeof(loaded_num_heads));
    file.read(reinterpret_cast<char*>(&loaded_num_layers), sizeof(loaded_num_layers));
    file.read(reinterpret_cast<char*>(&loaded_d_ff), sizeof(loaded_d_ff));
    file.read(reinterpret_cast<char*>(&loaded_max_seq_len), sizeof(loaded_max_seq_len));
    
    // Load output projection matrix
    size_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    
    if (rows == output_projection.rows && cols == output_projection.cols) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                file.read(reinterpret_cast<char*>(&output_projection[i][j]), sizeof(float));
            }
        }
    }
    
    file.close();
    std::cout << "Model loaded from " << filename << std::endl;
}

void TrainingTransformer::save_model_with_tokenizer(const std::string& filename, const SimpleTokenizer& tokenizer) const {
    // Save the model
    save_model(filename);
    
    // Save the tokenizer vocabulary
    std::string vocab_filename = filename + ".vocab";
    tokenizer.save_vocab(vocab_filename);
    
    std::cout << "Model and tokenizer saved to " << filename << " and " << vocab_filename << std::endl;
}

void TrainingTransformer::load_model_with_tokenizer(const std::string& filename, SimpleTokenizer& tokenizer) {
    // Load the model
    load_model(filename);
    
    // Load the tokenizer vocabulary
    std::string vocab_filename = filename + ".vocab";
    tokenizer.load_vocab(vocab_filename);
    
    std::cout << "Model and tokenizer loaded from " << filename << " and " << vocab_filename << std::endl;
}

void TrainingTransformer::set_learning_rate(float lr) {
    if (optimizer) {
        optimizer->set_learning_rate(lr);
    }
}

float TrainingTransformer::get_learning_rate() const {
    if (optimizer) {
        return optimizer->get_learning_rate();
    }
    return 0.0f;
}

// Training loop implementation
void train_model(TrainingConfig config) {
    std::cout << "=== Training Configuration ===" << std::endl;
    std::cout << "Epochs: " << config.epochs << std::endl;
    std::cout << "Batch size: " << config.batch_size << std::endl;
    std::cout << "Learning rate: " << config.learning_rate << std::endl;
    std::cout << "Model save path: " << config.model_save_path << std::endl;
    std::cout << std::endl;
    
    // Initialize tokenizer
    SimpleTokenizer tokenizer;
    
    // Create data loaders
    std::cout << "DEBUG: Tokenizer vocab size before data loading: " << tokenizer.vocab_size() << std::endl;
    DataLoader train_loader(config.train_data_path, config.batch_size, 64, tokenizer);
    DataLoader val_loader(config.val_data_path, config.batch_size, 64, tokenizer);
    std::cout << "DEBUG: Tokenizer vocab size after data loading: " << tokenizer.vocab_size() << std::endl;
    
    // Ensure tokenizer has a reasonable vocab size
    if (tokenizer.vocab_size() < 10) {
        // Pre-populate tokenizer with some basic words
        for (const auto& word : {"the", "and", "to", "of", "a", "in", "is", "it", "you", "that"}) {
            tokenizer.add_word(word);
        }
    }
    
    // Create model with correct vocab size
    size_t vocab_size = static_cast<size_t>(tokenizer.vocab_size());
    
    // DEBUG: Print detailed tokenizer info
    std::cout << "DEBUG: Tokenizer vocab size after data loading: " << tokenizer.vocab_size() << std::endl;
    std::cout << "DEBUG: Creating model with vocab size: " << vocab_size << std::endl;
    
    TrainingTransformer model(vocab_size, 128, 4, 2, 512, 64);
    model.set_learning_rate(config.learning_rate);
    
    std::cout << "Model created with vocab size: " << vocab_size << std::endl;
    std::cout << std::endl;
    
    // Training loop
    for (int epoch = 1; epoch <= config.epochs; ++epoch) {
        std::cout << "=== Epoch " << epoch << " ===" << std::endl;
        model.train_epoch(train_loader, val_loader, epoch);
        
        // Save model periodically
        if (epoch % 5 == 0 || epoch == config.epochs) {
            std::string save_path = config.model_save_path + "_epoch_" + std::to_string(epoch) + ".bin";
            model.save_model_with_tokenizer(save_path, tokenizer);
        }
        
        std::cout << std::endl;
    }
    
    // Save final model
    model.save_model_with_tokenizer(config.model_save_path, tokenizer);
    std::cout << "Training completed!" << std::endl;
}