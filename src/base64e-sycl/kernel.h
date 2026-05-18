typedef unsigned char uchar;

void base64_enc( const uchar* __restrict__ input,
                 uchar* __restrict__ output,
                 const char padCount,
                 const ulong numBlock,
                 const uint offset,
                 sycl::nd_item<1> &item)
{
    const uchar base64chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t id = item.get_global_id(0) + offset;
    if (id >= numBlock) return;
    // last implies there is padding for the last byte
    bool last = (padCount != 0) & (id == numBlock - 1);

    uchar si  = input[id*3];
    uchar si1 = input[id*3+1];
    uchar si2 = input[id*3+2];

    sycl::uchar4 result;
    result.x() = base64chars[si / 4]; // 6-bit rom
    result.y() = (last && padCount == 1) ? base64chars[(si * 16) % 64] : base64chars[(si * 16) % 64 + si1 / 16];
    result.z() = last ? ((padCount == 1) ? '=' : base64chars[(si1 * 4) % 64]) : base64chars[(si1 * 4) % 64 + si2 / 64];
    result.w() = last ? '=' : base64chars[si2 % 64];
    reinterpret_cast<sycl::uchar4 *>(output)[id] = result;
}

