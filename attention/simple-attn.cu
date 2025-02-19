#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>

/* This is a very simple implementation of self attention.
   There are no trainable weights in this implementation.
   We take word embedding matrix as our input and compute
   a context vector.
*/

#define NUM_WORDS 6
#define EMBED_DIM 3


__global__ void  attn_scores(float *input, float *attn_scores){
    // dot product between query and input rows

    int row = blockIdx.x;
    int col = threadIdx.x;
    
    float dot_prod = 0.0f;
    for(int d = 0; d < EMBED_DIM; d++){
        dot_prod += input[row * EMBED_DIM + d] * input[col * EMBED_DIM + d];
    }
    attn_scores[row * NUM_WORDS + col] = dot_prod;
}


// naive softmax for now
__global__ void softmax(float *attn_scores){
    int row = blockIdx.x;
    float norm = 0.0f;
    float max_val = -INFINITY;

    for(int i = 0; i < NUM_WORDS; i++){
        max_val = fmaxf(max_val, attn_scores[row * NUM_WORDS + i]);
    }

    for(int i = 0; i < NUM_WORDS; i++){
        attn_scores[row * NUM_WORDS + i] = expf(attn_scores[row * NUM_WORDS + i] - max_val);
        norm += attn_scores[row * NUM_WORDS + i];
    }

    for(int i = 0; i < NUM_WORDS; i++){
        attn_scores[row * NUM_WORDS + i] /=norm; // the scores now become weights
    }

}


__global__ void context_vector(float *input, float *attn_weights, float *context_vect){
    int row = blockIdx.x;
    int col = threadIdx.x;
    float context = 0.0f;

    // dot product of attention weights and each word embedding
    for(int i = 0; i < NUM_WORDS; i++){
        context += attn_weights[row * NUM_WORDS + i] * input[i * EMBED_DIM + col];
    }
    
    context_vect[row * EMBED_DIM + col] = context;
}


int main(){
    float token_emb[NUM_WORDS * EMBED_DIM] = {
        0.43, 0.15, 0.89,
        0.55, 0.87, 0.66,
        0.57, 0.85, 0.64,
        0.22, 0.58, 0.33,
        0.77, 0.25, 0.10,
        0.05, 0.80, 0.55
    };

    float *d_inputs, *d_attention_scores, *d_context_vectors;
    cudaMalloc(&d_inputs, NUM_WORDS * EMBED_DIM * sizeof(float));
    cudaMalloc(&d_attention_scores, NUM_WORDS * NUM_WORDS * sizeof(float));
    cudaMalloc(&d_context_vectors, NUM_WORDS * EMBED_DIM * sizeof(float));
    
    cudaMemcpy(d_inputs, token_emb, NUM_WORDS * EMBED_DIM * sizeof(float), cudaMemcpyHostToDevice);
    
    attn_scores<<<NUM_WORDS, NUM_WORDS>>>(d_inputs, d_attention_scores);
    cudaDeviceSynchronize();

    float h_attention_scores[NUM_WORDS * NUM_WORDS];
    cudaMemcpy(h_attention_scores, d_attention_scores, NUM_WORDS * NUM_WORDS * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Attention Scores Before Softmax:\n";
    for (int i = 0; i < NUM_WORDS; i++) {
        for (int j = 0; j < NUM_WORDS; j++) {
            std::cout << h_attention_scores[i * NUM_WORDS + j] << " ";
        }
        std::cout << "\n";
    }

    
    softmax<<<NUM_WORDS, 1>>>(d_attention_scores);
    cudaDeviceSynchronize();

    cudaMemcpy(h_attention_scores, d_attention_scores, NUM_WORDS * NUM_WORDS * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Attention Weights After Softmax:\n";
    for (int i = 0; i < NUM_WORDS; i++) {
        for (int j = 0; j < NUM_WORDS; j++) {
            std::cout << h_attention_scores[i * NUM_WORDS + j] << " ";
        }
        std::cout << "\n";
    }

    
    context_vector<<<NUM_WORDS, EMBED_DIM>>>(d_inputs, d_attention_scores, d_context_vectors);
    cudaDeviceSynchronize();
    
    float h_context_vectors[NUM_WORDS * EMBED_DIM];
    cudaMemcpy(h_context_vectors, d_context_vectors, NUM_WORDS * EMBED_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Context Vectors:\n";
    for (int i = 0; i < NUM_WORDS; i++) {
        std::cout << "(";
        for (int d = 0; d < EMBED_DIM; d++) {
            std::cout << h_context_vectors[i * EMBED_DIM + d] << (d < EMBED_DIM - 1 ? ", " : ")\n");
        }
    }
    
    cudaFree(d_inputs);
    cudaFree(d_attention_scores);
    cudaFree(d_context_vectors);
    
    return 0;
}




