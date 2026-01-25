inline void hadamard4x4a(float &p, float &q, float &r, float &s)
{
    float t = (p + q + r + s) * 0.5f;
    p = p - t;
    q = q - t;
    r = t - r;
    s = t - s;
}

inline void hadamard4x4b(float &p, float &q, float &r, float &s)
{
    float t = (p + q + r + s) * 0.5f;
    p = t - p;
    q = t - q;
    r = r - t;
    s = s - t;
}

void reference (
    const unsigned m_seed,
    float *globalPool,
    float *generatedRandomNumberPool,
    const float *chi2Corrections,
    int numBlocks
)
{
    const unsigned lcg_a = 241;
    const unsigned lcg_c = 59;
    const unsigned lcg_m = 256;
    const unsigned mod_mask = lcg_m - 1;

    for (int blockIdx = 0; blockIdx < numBlocks; ++blockIdx)
    {
        // same initial m_seed for each block
        unsigned t_seed = m_seed;

        // emulate shared memory
        float pool[WALLACE_POOL_SIZE + WALLACE_CHI2_SHARED_SIZE];

        unsigned offset = WALLACE_POOL_SIZE * blockIdx;

        // load global pool to shared pool
        for (unsigned threadIdx = 0; threadIdx < WALLACE_NUM_THREADS; ++threadIdx)
        {
            for (int i = 0; i < 8; ++i)
            {
                pool[threadIdx + WALLACE_NUM_THREADS * i] =
                    globalPool[offset + threadIdx + WALLACE_NUM_THREADS * i];
            }
        }

        // main generation
        for (unsigned loop = 0; loop < WALLACE_NUM_OUTPUTS_PER_RUN; ++loop)
        {
            t_seed = (1664525U * t_seed + 1013904223U);

            pool[WALLACE_CHI2_OFFSET] = chi2Corrections[blockIdx * WALLACE_NUM_OUTPUTS_PER_RUN + loop];

            float chi2CorrAndScale = pool[WALLACE_CHI2_OFFSET];

            for (unsigned threadIdx = 0; threadIdx < WALLACE_NUM_THREADS; ++threadIdx)
            {
                unsigned intermediate_address = loop * 8 * WALLACE_TOTAL_NUM_THREADS +
                    8 * WALLACE_NUM_THREADS * blockIdx + threadIdx;

                for (int i = 0; i < 8; ++i)
                {
                    generatedRandomNumberPool[intermediate_address + i * WALLACE_NUM_THREADS] =
                        pool[i * WALLACE_NUM_THREADS + threadIdx] * chi2CorrAndScale;
                }
            }
            // beware write after read race conditions when addresses of the pool are equal
            float rin0_0[WALLACE_NUM_THREADS],
                  rin1_0[WALLACE_NUM_THREADS],
                  rin2_0[WALLACE_NUM_THREADS],
                  rin3_0[WALLACE_NUM_THREADS],
                  rin0_1[WALLACE_NUM_THREADS],
                  rin1_1[WALLACE_NUM_THREADS],
                  rin2_1[WALLACE_NUM_THREADS],
                  rin3_1[WALLACE_NUM_THREADS];
            for (unsigned pass = 0; pass < WALLACE_NUM_POOL_PASSES; ++pass)
            {
                for (unsigned threadIdx = 0; threadIdx < WALLACE_NUM_THREADS; ++threadIdx)
                {
                    unsigned seed = (t_seed + threadIdx) & mod_mask;

                    seed = (lcg_a * seed + lcg_c) & mod_mask;
                    rin0_0[threadIdx] = pool[(seed << 3) + 0];

                    seed = (lcg_a * seed + lcg_c) & mod_mask;
                    rin1_0[threadIdx] = pool[(seed << 3) + 1];

                    seed = (lcg_a * seed + lcg_c) & mod_mask;
                    rin2_0[threadIdx] = pool[(seed << 3) + 2];

                    seed = (lcg_a * seed + lcg_c) & mod_mask;
                    rin3_0[threadIdx] = pool[(seed << 3) + 3];

                    seed = (lcg_a * seed + lcg_c) & mod_mask;
                    rin0_1[threadIdx] = pool[(seed << 3) + 4];

                    seed = (lcg_a * seed + lcg_c) & mod_mask;
                    rin1_1[threadIdx] = pool[(seed << 3) + 5];

                    seed = (lcg_a * seed + lcg_c) & mod_mask;
                    rin2_1[threadIdx] = pool[(seed << 3) + 6];

                    seed = (lcg_a * seed + lcg_c) & mod_mask;
                    rin3_1[threadIdx] = pool[(seed << 3) + 7];

                    hadamard4x4a(rin0_0[threadIdx], rin1_0[threadIdx], rin2_0[threadIdx], rin3_0[threadIdx]);
                }
                // write to pool after read from pool
                for (unsigned threadIdx = 0; threadIdx < WALLACE_NUM_THREADS; ++threadIdx)
                {
                    pool[0 * WALLACE_NUM_THREADS + threadIdx] = rin0_0[threadIdx];
                    pool[1 * WALLACE_NUM_THREADS + threadIdx] = rin1_0[threadIdx];
                    pool[2 * WALLACE_NUM_THREADS + threadIdx] = rin2_0[threadIdx];
                    pool[3 * WALLACE_NUM_THREADS + threadIdx] = rin3_0[threadIdx];

                    hadamard4x4b(rin0_1[threadIdx], rin1_1[threadIdx], rin2_1[threadIdx], rin3_1[threadIdx]);
                    pool[4 * WALLACE_NUM_THREADS + threadIdx] = rin0_1[threadIdx];
                    pool[5 * WALLACE_NUM_THREADS + threadIdx] = rin1_1[threadIdx];
                    pool[6 * WALLACE_NUM_THREADS + threadIdx] = rin2_1[threadIdx];
                    pool[7 * WALLACE_NUM_THREADS + threadIdx] = rin3_1[threadIdx];
                }
            }
        }

        for (unsigned threadIdx = 0; threadIdx < WALLACE_NUM_THREADS; ++threadIdx)
        {
            for (int i = 0; i < 8; ++i)
            {
                globalPool[offset + threadIdx + WALLACE_NUM_THREADS * i] =
                    pool[threadIdx + WALLACE_NUM_THREADS * i];
            }
        }
    }
}

