#include "avar_gpu.cuh"
#include "macros.cuh"
#include <stdio.h>

#define WARP_SIZE 32
#define WORD_SIZE 32

__global__ void avar_compress_gpu (avar_header comp_h, int *data, int *compressed_data, unsigned long length)
{
    int warp_th = (threadIdx.x % WARP_SIZE); 
    unsigned long pos = blockIdx.x * blockDim.x + threadIdx.x - warp_th;
    unsigned long data_id = pos * WARP_SIZE + warp_th;
    unsigned long cdata_id = pos * comp_h.bit_length + warp_th;
    avar_compress_base_gpu(comp_h, data_id, cdata_id, data, compressed_data, length);
}

__global__ void avar_decompress_gpu (avar_header comp_h, int *compressed_data, int * decompress_data, unsigned long length)
{
    int warp_th = (threadIdx.x % WARP_SIZE); 
    unsigned long pos = blockIdx.x * blockDim.x + threadIdx.x - warp_th;
    unsigned long data_id = pos * WARP_SIZE + warp_th;
    unsigned long cdata_id = pos * comp_h.bit_length + warp_th;
    avar_decompress_base_gpu(comp_h, cdata_id, data_id, compressed_data, decompress_data, length);
}

__host__ void run_avar_compress_gpu(avar_header comp_h, int *data, int *compressed_data, unsigned long length)
{
    int block_size = WARP_SIZE * 8; // better occupancy 
    unsigned long block_number = (length + block_size * WARP_SIZE - 1) / (block_size * WARP_SIZE);
    avar_compress_gpu <<<block_number, block_size>>> (comp_h, data, compressed_data, length);
}

__host__ void run_avar_decompress_gpu(avar_header comp_h, int *compressed_data, int *data, unsigned long length)
{
    int block_size = WARP_SIZE * 8; // better occupancy
    unsigned long block_number = (length + block_size * WARP_SIZE - 1) / (block_size * WARP_SIZE);
    avar_decompress_gpu <<<block_number, block_size>>> (comp_h, compressed_data, data, length);
}

__device__  void avar_compress_base_gpu (avar_header comp_h, unsigned long data_id, unsigned long comp_data_id, int *data, int *compressed_data, unsigned long length)
{
    int v1, value = 0;
    unsigned int v1_pos=0, v1_len;
    unsigned long pos=comp_data_id, pos_data=data_id;

    for (unsigned int i = 0; i < WARP_SIZE && pos_data < length; ++i)
    {
        v1 = data[pos_data];
        pos_data += WARP_SIZE;

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
    if (pos_data >= length  && pos_data < length + WARP_SIZE)
    {
        compressed_data[pos] = value;
    }
}

__device__ void avar_decompress_base_gpu (avar_header comp_h, unsigned long comp_data_id, unsigned long data_id, int *compressed_data, int *data, unsigned long length)
{
    unsigned long pos=comp_data_id, pos_decomp=data_id;
    unsigned int v1_pos=0, v1_len;
    int v1, ret;

    if (pos_decomp > length ) // Decompress not more elements then length
        return;
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
