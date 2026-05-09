// ============================================================
// CPU reference  (BGMV-Shrink only, for correctness check)
// ============================================================
void ref_bgmv_shrink_cpu(
    float*        output,       // [num_tokens, lora_rank]
    const float*  input,        // [num_tokens, hidden_size]  (fp32 copy)
    const float*  weights,      // [num_loras, lora_rank, hidden_size]
    const int*    lora_indices, // [num_tokens]
    int num_tokens, int hidden_size, int lora_rank, float scaling)
{
    for (int t = 0; t < num_tokens; t++) {
        int lid = lora_indices[t];
        for (int r = 0; r < lora_rank; r++) {
            float acc = 0.f;
            for (int k = 0; k < hidden_size; k++)
                acc += input[t * hidden_size + k]
                     * weights[lid * lora_rank * hidden_size + r * hidden_size + k];
            output[t * lora_rank + r] += scaling * acc;
        }
    }
}

void ref_bgmv_expand_cpu(
    float*        output,       // [num_tokens, hidden_size]
    const float*  input,        // [num_tokens, lora_rank]  (fp32 copy)
    const float*  weights,      // [num_loras, hidden_size, lora_rank]
    const int*    lora_indices, // [num_tokens]
    int num_tokens, int hidden_size, int lora_rank, bool add_inputs)
{
    for (int t = 0; t < num_tokens; t++) {
        int lid = lora_indices[t];
        for (int r = 0; r < hidden_size; r++) {
            float acc = 0.f;
            for (int k = 0; k < lora_rank; k++)
                acc += input[t * lora_rank + k]
                     * weights[lid * lora_rank * hidden_size + r * lora_rank + k];
            float p = add_inputs ? output[t * hidden_size + r] : 0;
            output[t * hidden_size + r] = p + acc;
        }
    }
}


