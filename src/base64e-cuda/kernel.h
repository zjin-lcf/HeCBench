typedef unsigned char uchar;

__global__
void base64_enc( const uchar* __restrict__ input,
                 uchar* __restrict__ output,
                 const char padCount,
                 const ulong numBlock,
                 const uint offset)
{
    const uchar base64chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t id = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (id >= numBlock) return;
    // last implies there is padding for the last byte
    bool last = (padCount != 0) & (id == numBlock - 1);

    //const uchar3 loaded = vload3(id, input);
    uchar3 loaded = reinterpret_cast<const uchar3*>(input)[id];
    uchar si  = loaded.x;
    uchar si1 = loaded.y;
    uchar si2 = loaded.z;

    uchar4 result;
    result.x = base64chars[si / 4]; // 6-bit rom
    result.y = (last && padCount == 1) ? base64chars[(si * 16) % 64] : base64chars[(si * 16) % 64 + si1 / 16];
    result.z = last ? ((padCount == 1) ? '=' : base64chars[(si1 * 4) % 64]) : base64chars[(si1 * 4) % 64 + si2 / 64];
    result.w = last ? '=' : base64chars[si2 % 64];
    reinterpret_cast<uchar4*>(output)[id] = result;
}

