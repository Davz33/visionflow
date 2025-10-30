//
// Custom Metal Shaders for M4 Optimization
// Optimized tensor operations for video generation
//

#include <metal_stdlib>
using namespace metal;

// Optimized matrix multiplication kernel for attention mechanisms
kernel void optimized_matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Use threadgroup memory for better cache utilization
    constexpr uint tile_size = 16;
    threadgroup float Asub[tile_size][tile_size];
    threadgroup float Bsub[tile_size][tile_size];
    
    uint row = gid.y * tile_size;
    uint col = gid.x * tile_size;
    
    float sum[tile_size][tile_size];
    for (uint i = 0; i < tile_size; i++) {
        for (uint j = 0; j < tile_size; j++) {
            sum[i][j] = 0.0;
        }
    }
    
    // Tiled matrix multiplication
    for (uint tile = 0; tile < (K + tile_size - 1) / tile_size; tile++) {
        // Load tiles into shared memory
        uint tile_start = tile * tile_size;
        
        // Load A tile
        for (uint i = 0; i < tile_size; i++) {
            for (uint j = 0; j < tile_size; j++) {
                uint ai = row + i;
                uint aj = tile_start + j;
                Asub[i][j] = (ai < M && aj < K) ? A[ai * K + aj] : 0.0;
            }
        }
        
        // Load B tile
        for (uint i = 0; i < tile_size; i++) {
            for (uint j = 0; j < tile_size; j++) {
                uint bi = tile_start + i;
                uint bj = col + j;
                Bsub[i][j] = (bi < K && bj < N) ? B[bi * N + bj] : 0.0;
            }
        }
        
        // Compute partial sum
        for (uint i = 0; i < tile_size; i++) {
            for (uint j = 0; j < tile_size; j++) {
                for (uint k = 0; k < tile_size; k++) {
                    sum[i][j] += Asub[i][k] * Bsub[k][j];
                }
            }
        }
    }
    
    // Write results
    for (uint i = 0; i < tile_size; i++) {
        for (uint j = 0; j < tile_size; j++) {
            uint ci = row + i;
            uint cj = col + j;
            if (ci < M && cj < N) {
                C[ci * N + cj] = sum[i][j];
            }
        }
    }
}

// Optimized attention computation kernel
kernel void scaled_dot_product_attention_optimized(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.y;
    uint head_idx = gid.x;
    uint seq_idx = gid.z;
    
    if (seq_idx >= seq_len) return;
    
    // Compute QK^T
    float score = 0.0;
    for (uint i = 0; i < head_dim; i++) {
        uint q_idx = batch_idx * seq_len * head_dim + seq_idx * head_dim + i;
        uint k_idx = batch_idx * seq_len * head_dim + seq_idx * head_dim + i;
        score += Q[q_idx] * K[k_idx];
    }
    score *= scale;
    
    // Apply mask if provided
    if (mask != nullptr) {
        uint mask_idx = batch_idx * seq_len * seq_len + seq_idx * seq_len + seq_idx;
        score += mask[mask_idx];
    }
    
    // Softmax (simplified - full softmax would need reduction)
    // For now, store raw scores
    uint out_idx = batch_idx * seq_len * head_dim + seq_idx * head_dim + head_idx;
    output[out_idx] = score;
}

// Optimized convolution kernel for video frames
kernel void optimized_conv2d(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant float* weights [[buffer(0)]],
    constant uint& kernel_size [[buffer(1)]],
    constant uint& stride [[buffer(2)]],
    constant uint& padding [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    float sum = 0.0;
    
    // Sample from input texture with kernel
    for (uint ky = 0; ky < kernel_size; ky++) {
        for (uint kx = 0; kx < kernel_size; kx++) {
            int2 coord = int2(gid) * int2(stride) + int2(kx, ky) - int2(padding);
            if (coord.x >= 0 && coord.y >= 0 && coord.x < input.get_width() && coord.y < input.get_height()) {
                float val = input.read(uint2(coord)).r;
                float weight = weights[ky * kernel_size + kx];
                sum += val * weight;
            }
        }
    }
    
    output.write(float4(sum, sum, sum, 1.0), gid);
}

// Optimized element-wise operations
kernel void elementwise_add(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    C[gid] = A[gid] + B[gid];
}

kernel void elementwise_multiply(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    C[gid] = A[gid] * B[gid];
}

// Optimized normalization kernel
kernel void layer_norm_optimized(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant uint& feature_dim [[buffer(4)]],
    constant uint& num_features [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint feature_idx = gid;
    if (feature_idx >= num_features) return;
    
    // Compute mean
    float mean = 0.0;
    for (uint i = 0; i < feature_dim; i++) {
        mean += input[feature_idx * feature_dim + i];
    }
    mean /= feature_dim;
    
    // Compute variance
    float variance = 0.0;
    for (uint i = 0; i < feature_dim; i++) {
        float diff = input[feature_idx * feature_dim + i] - mean;
        variance += diff * diff;
    }
    variance /= feature_dim;
    
    // Normalize
    float inv_std = 1.0 / sqrt(variance + 1e-5);
    for (uint i = 0; i < feature_dim; i++) {
        uint idx = feature_idx * feature_dim + i;
        float normalized = (input[idx] - mean) * inv_std;
        output[idx] = normalized * gamma[i] + beta[i];
    }
}
