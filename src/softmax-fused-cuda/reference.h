template<typename T>
T* make_random(size_t N) {
    T* arr = (T*)aligned_alloc(1024, N * sizeof(T));
    for (size_t i = 0; i < N; i++) {
        arr[i] = (rand() / (float)RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

int* make_random_int(size_t N, int V) {
    int* arr = (int*)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    return arr;
}

template <typename input_t, typename output_t, typename acc_t>
void reference (output_t* out, const input_t* inp, const uint8_t* mask, acc_t scale, 
                int pad_batch, int batch, int head, int seq_len, int C) {
    // inp is (N, C), each row of inp will get softmaxed
    acc_t *tmp_row = (acc_t*) malloc (sizeof(acc_t) * C);

    const uint8_t* mask_row;
    for (int b = 0; b < batch; b++) {
    for (int h = 0; h < head; h++) {
    for (int s = 0; s < seq_len; s++) {
        size_t i = (uint64_t)b * head * seq_len + h * seq_len + s;
        const input_t* inp_row = inp + i * C;
             output_t* out_row = out + i * C;

        if (pad_batch != 1)
           mask_row = mask + (b * seq_len + s) * C;
        else
           mask_row = mask + s * C;

        for (int j = 0; j < C; j++) {
            if (mask_row[j] != 1) {
                tmp_row[j] = (acc_t)inp_row[j] * scale;
            } else {
                tmp_row[j] = (acc_t)-10000.0;
            }
            //printf("ref scale: N=%zu j=%d %f\n", i, j, tmp_row[j]);
        }

        //acc_t maxval = -INFINITY;
        acc_t maxval = -10000.0;
        for (int j = 0; j < C; j++) {
            if (tmp_row[j] > maxval) {
                maxval = tmp_row[j];
            }
        }
        //printf("ref maxval: N=%zu %f\n", i, maxval);
        uint8_t mask_value = (maxval == (acc_t)-10000.0) ? 1 : 0;
        acc_t sum = 0.0;
        for (int j = 0; j < C; j++) {
            tmp_row[j] = expf(tmp_row[j] - maxval);
            sum += tmp_row[j];
        }
        for (int j = 0; j < C; j++) {
            if (mask_value) 
                out_row[j] = (output_t)0;
            else
                out_row[j] = tmp_row[j] / sum;
            //printf("ref out: N=%zu j=%d %f %f\n", i, j, tmp_row[j], sum);
        }
    } } }
    free(tmp_row);
}
