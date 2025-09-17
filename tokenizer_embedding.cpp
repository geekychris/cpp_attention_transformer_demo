#include "transformer.h"
#include <sstream>
#include <algorithm>
#include <fstream>

// Define static const members
const int SimpleTokenizer::PAD_TOKEN;
const int SimpleTokenizer::UNK_TOKEN;
const int SimpleTokenizer::BOS_TOKEN;
const int SimpleTokenizer::EOS_TOKEN;

// Simple tokenizer implementation
SimpleTokenizer::SimpleTokenizer() : next_id(4) {  // Start after special tokens
    // Initialize special tokens
    word_to_id["<PAD>"] = PAD_TOKEN;
    word_to_id["<UNK>"] = UNK_TOKEN;
    word_to_id["<BOS>"] = BOS_TOKEN;
    word_to_id["<EOS>"] = EOS_TOKEN;
    
    id_to_word[PAD_TOKEN] = "<PAD>";
    id_to_word[UNK_TOKEN] = "<UNK>";
    id_to_word[BOS_TOKEN] = "<BOS>";
    id_to_word[EOS_TOKEN] = "<EOS>";
}

void SimpleTokenizer::add_word(const std::string& word) {
    if (word_to_id.find(word) == word_to_id.end()) {
        word_to_id[word] = next_id;
        id_to_word[next_id] = word;
        ++next_id;
    }
}

std::vector<int> SimpleTokenizer::encode(const std::string& text) {
    std::vector<int> tokens;
    tokens.push_back(BOS_TOKEN);  // Start of sequence
    
    // Simple whitespace tokenization
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        // Convert to lowercase and remove punctuation (very basic)
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        
        // Remove basic punctuation
        word.erase(std::remove_if(word.begin(), word.end(), [](char c) {
            return c == '.' || c == ',' || c == '!' || c == '?' || c == ';' || c == ':';
        }), word.end());
        
        if (!word.empty()) {
            // Add word to vocabulary if not seen before
            add_word(word);
            tokens.push_back(word_to_id[word]);
        }
    }
    
    tokens.push_back(EOS_TOKEN);  // End of sequence
    return tokens;
}

std::string SimpleTokenizer::decode(const std::vector<int>& tokens) {
    std::ostringstream oss;
    bool first = true;
    
    for (int token_id : tokens) {
        if (token_id == BOS_TOKEN || token_id == EOS_TOKEN || token_id == PAD_TOKEN) {
            continue;  // Skip special tokens in output
        }
        
        if (!first) {
            oss << " ";
        }
        
        if (id_to_word.find(token_id) != id_to_word.end()) {
            oss << id_to_word.at(token_id);
        } else {
            oss << "<UNK>";
        }
        first = false;
    }
    
    return oss.str();
}

void SimpleTokenizer::save_vocab(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open vocab file for saving: " << filename << std::endl;
        return;
    }
    
    // Save next_id first
    file << next_id << std::endl;
    
    // Save all word->id mappings
    for (const auto& pair : word_to_id) {
        file << pair.first << " " << pair.second << std::endl;
    }
    
    file.close();
}

void SimpleTokenizer::load_vocab(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open vocab file for loading: " << filename << std::endl;
        return;
    }
    
    // Clear existing vocabulary
    word_to_id.clear();
    id_to_word.clear();
    
    // Load next_id
    file >> next_id;
    
    // Load all word->id mappings
    std::string word;
    int id;
    while (file >> word >> id) {
        word_to_id[word] = id;
        id_to_word[id] = word;
    }
    
    file.close();
}

// Embedding layer implementation
Embedding::Embedding(size_t vocab_sz, size_t model_dim) 
    : embeddings(vocab_sz, model_dim), vocab_size(vocab_sz), d_model(model_dim) {
    initialize_weights();
}

void Embedding::initialize_weights() {
    // Initialize embeddings with normal distribution
    float std = std::sqrt(1.0f / d_model);
    embeddings.randomize(std);
}

Matrix Embedding::forward(const std::vector<int>& tokens) const {
    size_t seq_len = tokens.size();
    Matrix output(seq_len, d_model);
    
    for (size_t i = 0; i < seq_len; ++i) {
        int token_id = tokens[i];
        if (token_id >= 0 && token_id < static_cast<int>(vocab_size) && 
            static_cast<size_t>(token_id) < embeddings.rows) {
            // Copy the embedding vector for this token
            for (size_t j = 0; j < d_model && j < embeddings.cols; ++j) {
                output[i][j] = embeddings[token_id][j];
            }
        } else {
            // Unknown token - use zero vector or UNK embedding
            int unk_token = SimpleTokenizer::UNK_TOKEN;
            if (unk_token >= 0 && static_cast<size_t>(unk_token) < embeddings.rows) {
                for (size_t j = 0; j < d_model && j < embeddings.cols; ++j) {
                    output[i][j] = embeddings[unk_token][j];
                }
            } else {
                // Fallback: use zero vector
                for (size_t j = 0; j < d_model; ++j) {
                    output[i][j] = 0.0f;
                }
            }
        }
    }
    
    // Scale embeddings by sqrt(d_model) as in original paper
    float scale = std::sqrt(static_cast<float>(d_model));
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < d_model; ++j) {
            output[i][j] *= scale;
        }
    }
    
    return output;
}