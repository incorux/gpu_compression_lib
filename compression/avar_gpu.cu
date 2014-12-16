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

#define WARP_SIZE 32
#define WORD_SIZE 32

__global__ void avar_compress_gpu (avar_header comp_h, int *data, int *compressed_data, size_t length)
{
    long warp_th = (threadIdx.x % WARP_SIZE); 
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x - warp_th;
    size_t data_id = pos * WARP_SIZE + warp_th;
    size_t cdata_id = pos * comp_h.bit_length + warp_th;
    avar_compress_base_gpu(comp_h, data_id, cdata_id, data, compressed_data, length);
}

__global__ void avar_decompress_gpu (avar_header comp_h, int *compressed_data, int * decompress_data, size_t length)
{
    int warp_th = (threadIdx.x % WARP_SIZE); 
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x - warp_th;
    size_t data_id = pos * WARP_SIZE + warp_th;
    size_t cdata_id = pos * comp_h.bit_length + warp_th;
    avar_decompress_base_gpu(comp_h, cdata_id, data_id, compressed_data, decompress_data, length);
}


__host__ void run_avar_compress_gpu(avar_header comp_h, int *data, int *compressed_data, size_t length)
{
    int block_size = WARP_SIZE * 8; // better occupancy 
    size_t block_number = (length + block_size * WARP_SIZE - 1) / (block_size * WARP_SIZE);
    avar_compress_gpu <<<block_number, block_size>>> (comp_h, data, compressed_data, length);
}

__host__ void run_avar_decompress_gpu(avar_header comp_h, int *compressed_data, int *data, size_t length)
{
    int block_size = WARP_SIZE * 8; // better occupancy
    size_t block_number = (length + block_size * WARP_SIZE - 1) / (block_size * WARP_SIZE);
    avar_decompress_gpu <<<block_number, block_size>>> (comp_h, compressed_data, data, length);
}


__device__  void avar_compress_base_gpu (avar_header comp_h, size_t data_id, size_t comp_data_id, int *data, int *compressed_data, size_t length)
{
    int v1, value = 0;
    unsigned int v1_pos=0, v1_len;
    size_t pos=comp_data_id, pos_decomp=data_id;

    for (unsigned int i = 0; i < WARP_SIZE && pos_decomp < length; ++i)
    {
        v1 = data[pos_decomp];
        pos_decomp += WARP_SIZE;

        if (v1_pos + comp_h.bit_length >= WORD_SIZE){
            v1_len = WORD_SIZE - v1_pos;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);

            compressed_data[pos] = value;

            v1_pos = comp_h.bit_length - v1_len;
            value = GETNPBITS(v1, v1_pos, v1_len); 

            pos += WARP_SIZE;  
        } else {
            v1_len = comp_h.bit_length;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);
            v1_pos += v1_len;
        }
    }
    if (pos_decomp >= length  && pos_decomp < length + WARP_SIZE)
    {
        compressed_data[pos] = value;
    }
}

__device__ void avar_decompress_base_gpu (avar_header comp_h, size_t comp_data_id, size_t data_id, int *compressed_data, int *data, size_t length)
{
    size_t pos=comp_data_id, pos_decomp=data_id;
    unsigned int v1_pos=0, v1_len;
    int v1, ret;

    v1 = compressed_data[pos];
    for (unsigned int i = 0; i < WARP_SIZE && pos_decomp < length; ++i)
    {
        if (v1_pos + comp_h.bit_length >= WORD_SIZE){ 
            v1_len = WORD_SIZE - v1_pos;
            ret = GETNPBITS(v1, v1_len, v1_pos);

            pos += WARP_SIZE;  
            v1 = compressed_data[pos];

            v1_pos = comp_h.bit_length - v1_len;
            ret = ret | ((GETNBITS(v1, v1_pos))<< v1_len);
        } else {
            v1_len = comp_h.bit_length;
            ret = GETNPBITS(v1, v1_len, v1_pos);
            v1_pos += v1_len;
        }

        data[pos_decomp] = ret;
        pos_decomp += WARP_SIZE;
    }
}
