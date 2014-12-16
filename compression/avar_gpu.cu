#include "avar_gpu.cuh"
#include <stdio.h>


#if __CUDA_ARCH__ > 120  // This improves performance 
__device__ __inline__ unsigned int GETNPBITS(
  int source,
  unsigned int num_bits,
  unsigned int bit_start)
 {
  unsigned int bits;
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"((unsigned int) source), "r"(bit_start), "r"(num_bits));
  return bits;
 }
#define GETNBITS(X,N) GETNPBITS(X,N,0)
#else
#include "macros.cuh"
#endif

__global__ void avar_compress_gpu (avar_header comp_h, int *data, int *compressed_data, long length)
{
    long warp_th = (threadIdx.x & 31); 
    long pos = blockIdx.x * blockDim.x + threadIdx.x - warp_th;
    long data_id = pos * 32 + warp_th;
    long cdata_id = pos * comp_h.bit_length + warp_th;
    avar_compress_base_gpu(comp_h, data_id, cdata_id, data, compressed_data, length);
}

__global__ void avar_decompress_gpu (avar_header comp_h, int *compressed_data, int * decompress_data, long length)
{
    long warp_th = (threadIdx.x & 31); 
    long pos = blockIdx.x * blockDim.x + threadIdx.x - warp_th;
    long data_id = pos * 32 + warp_th;
    long cdata_id = pos * comp_h.bit_length + warp_th;
    avar_decompress_base_gpu(comp_h, cdata_id, data_id, compressed_data, decompress_data, length);
}


__host__ void run_avar_compress_gpu(avar_header comp_h, int *data, int *compressed_data, long length)
{
    int block_size = 32 * 8;
    long block_number = (length + block_size * comp_h.warp_size - 1) / (block_size * comp_h.warp_size);
    avar_compress_gpu <<<block_number, block_size>>> (comp_h, data, compressed_data, length);
}

__host__ void run_avar_decompress_gpu(avar_header comp_h, int *compressed_data, int *data, long length)
{
    int block_size = 32 * 8;
    long block_number = (length + block_size * comp_h.warp_size - 1) / (block_size * comp_h.warp_size);
    avar_decompress_gpu <<<block_number, block_size>>> (comp_h, compressed_data, data, length);
}


__device__  void avar_compress_base_gpu (avar_header comp_h, long data_id, long comp_data_id, int *data, int *compressed_data, long length)
{
    int v1, value = 0;
    unsigned int v1_pos=0, v1_len;
    long pos=comp_data_id, pos_decomp=data_id;

    /*#pragma unroll*/
    for (unsigned int i = 0; i < 32 && pos_decomp < length; ++i)
    {
        v1 = data[pos_decomp];
        pos_decomp += comp_h.warp_size;

        if (v1_pos + comp_h.bit_length >= comp_h.word_size){
            v1_len = comp_h.word_size - v1_pos;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);

            compressed_data[pos] = value;

            v1_pos = comp_h.bit_length - v1_len;
            value = GETNPBITS(v1, v1_pos, v1_len); 

            pos += comp_h.warp_size;  
        } else {
            v1_len = comp_h.bit_length;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);
            v1_pos += v1_len;
        }
    }
    if (pos_decomp >= length  && pos_decomp < length + 32)
    {
        compressed_data[pos] = value;
    }
}

__device__ void avar_decompress_base_gpu (avar_header comp_h, long comp_data_id, long data_id, int *compressed_data, int *data, long length)
{
    long pos=comp_data_id, pos_decomp=data_id;
    unsigned int v1_pos=0, v1_len;
    int v1, ret;

    v1 = compressed_data[pos];
    /*#pragma unroll*/
    for (unsigned int i = 0; i < 32 && pos_decomp < length; ++i)
    {
        if (v1_pos + comp_h.bit_length >= comp_h.word_size){ 
            v1_len = comp_h.word_size - v1_pos;
            ret = GETNPBITS(v1, v1_len, v1_pos);

            pos += comp_h.warp_size;  
            v1 = compressed_data[pos];

            v1_pos = comp_h.bit_length - v1_len;
            ret = ret | ((GETNBITS(v1, v1_pos))<< v1_len);
        } else {
            v1_len = comp_h.bit_length;
            ret = GETNPBITS(v1, v1_len, v1_pos);
            v1_pos += v1_len;
        }

        data[pos_decomp] = ret;
        pos_decomp += comp_h.warp_size;
    }
}
