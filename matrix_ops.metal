#include <metal_stdlib>
using namespace metal;

// Matrix addition kernel
kernel void matrix_add(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* result [[buffer(2)]],
                      constant uint& rows [[buffer(3)]],
                      constant uint& cols [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    
    if (gid.x >= cols || gid.y >= rows) return;
    
    uint index = gid.y * cols + gid.x;
    result[index] = a[index] + b[index];
}

// Matrix transpose kernel
kernel void matrix_transpose(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint& rows [[buffer(2)]],
                           constant uint& cols [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    
    if (gid.x >= cols || gid.y >= rows) return;
    
    uint input_index = gid.y * cols + gid.x;
    uint output_index = gid.x * rows + gid.y;
    
    output[output_index] = input[input_index];
}

// Matrix randomization kernel (Box-Muller transform for normal distribution)
kernel void matrix_randomize(device float* matrix [[buffer(0)]],
                            device const float* random_values [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            constant float& std_dev [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    
    if (gid >= size) return;
    
    // Use pre-generated random values and apply Box-Muller transform
    if (gid % 2 == 0 && gid + 1 < size) {
        float u1 = random_values[gid];
        float u2 = random_values[gid + 1];
        
        // Box-Muller transform
        float z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI_F * u2);
        float z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI_F * u2);
        
        matrix[gid] = z0 * std_dev;
        if (gid + 1 < size) {
            matrix[gid + 1] = z1 * std_dev;
        }
    }
}

// Optimized matrix multiplication kernel (tiled approach)
kernel void matrix_multiply_tiled(device const float* a [[buffer(0)]],
                                 device const float* b [[buffer(1)]],
                                 device float* c [[buffer(2)]],
                                 constant uint& M [[buffer(3)]], // rows of A
                                 constant uint& N [[buffer(4)]], // cols of B
                                 constant uint& K [[buffer(5)]], // cols of A / rows of B
                                 threadgroup float* a_shared [[threadgroup(0)]],
                                 threadgroup float* b_shared [[threadgroup(1)]],
                                 uint2 tgid [[threadgroup_position_in_grid]],
                                 uint2 tid [[thread_position_in_threadgroup]],
                                 uint2 tg_size [[threads_per_threadgroup]]) {
    
    const uint TILE_SIZE = 16; // Adjust based on GPU capabilities
    
    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;
    
    float sum = 0.0;
    
    // Loop over tiles
    for (uint tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        
        // Load tile of A into shared memory
        uint a_row = row;
        uint a_col = tile * TILE_SIZE + tid.x;
        if (a_row < M && a_col < K) {
            a_shared[tid.y * TILE_SIZE + tid.x] = a[a_row * K + a_col];
        } else {
            a_shared[tid.y * TILE_SIZE + tid.x] = 0.0;
        }
        
        // Load tile of B into shared memory
        uint b_row = tile * TILE_SIZE + tid.y;
        uint b_col = col;
        if (b_row < K && b_col < N) {
            b_shared[tid.y * TILE_SIZE + tid.x] = b[b_row * N + b_col];
        } else {
            b_shared[tid.y * TILE_SIZE + tid.x] = 0.0;
        }
        
        // Synchronize threads
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial sum for this tile
        for (uint k = 0; k < TILE_SIZE; ++k) {
            sum += a_shared[tid.y * TILE_SIZE + k] * b_shared[k * TILE_SIZE + tid.x];
        }
        
        // Synchronize before next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}

// Element-wise operations
kernel void matrix_scale(device float* matrix [[buffer(0)]],
                        constant float& scale [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    
    if (gid >= size) return;
    matrix[gid] *= scale;
}

// GELU activation function
kernel void matrix_gelu(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& size [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    
    if (gid >= size) return;
    
    float x = input[gid];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    float cube = x * x * x;
    float inner = sqrt(2.0 / M_PI_F) * (x + 0.044715 * cube);
    output[gid] = 0.5 * x * (1.0 + tanh(inner));
}

// Softmax kernel (numerically stable version)
kernel void matrix_softmax_rows(device const float* input [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               constant uint& rows [[buffer(2)]],
                               constant uint& cols [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
    
    if (gid >= rows) return;
    
    uint row_start = gid * cols;
    
    // Find maximum value in the row for numerical stability
    float max_val = input[row_start];
    for (uint j = 1; j < cols; ++j) {
        max_val = max(max_val, input[row_start + j]);
    }
    
    // Compute exponentials and sum
    float sum = 0.0;
    for (uint j = 0; j < cols; ++j) {
        float exp_val = exp(input[row_start + j] - max_val);
        output[row_start + j] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for (uint j = 0; j < cols; ++j) {
        output[row_start + j] /= sum;
    }
}

// Layer normalization kernel
kernel void layer_norm(device const float* input [[buffer(0)]],
                      device float* output [[buffer(1)]],
                      device const float* gamma [[buffer(2)]],
                      device const float* beta [[buffer(3)]],
                      constant uint& rows [[buffer(4)]],
                      constant uint& cols [[buffer(5)]],
                      constant float& eps [[buffer(6)]],
                      uint gid [[thread_position_in_grid]]) {
    
    if (gid >= rows) return;
    
    uint row_start = gid * cols;
    
    // Compute mean
    float mean = 0.0;
    for (uint j = 0; j < cols; ++j) {
        mean += input[row_start + j];
    }
    mean /= cols;
    
    // Compute variance
    float variance = 0.0;
    for (uint j = 0; j < cols; ++j) {
        float diff = input[row_start + j] - mean;
        variance += diff * diff;
    }
    variance /= cols;
    
    // Normalize and apply scale/shift
    float inv_std = 1.0 / sqrt(variance + eps);
    for (uint j = 0; j < cols; ++j) {
        float normalized = (input[row_start + j] - mean) * inv_std;
        output[row_start + j] = normalized * gamma[j] + beta[j];
    }
}