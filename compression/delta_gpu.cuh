#ifndef DELTA_CUH_KKZZHX97
#define DELTA_CUH_KKZZHX97

template <typename T> __global__ void delta_compress_gpu (T *data, T *compressed_data, T *spoints, unsigned int bit_length, unsigned long length, unsigned long spoints_length);

template <typename T> __global__ void delta_decompress_gpu (T *compressed_data, T *spoints, T *data, unsigned long length, unsigned int spoints_length, int width=32);
#endif /* end of include guard: DELTA_CUH_KKZZHX97 */
