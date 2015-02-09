#include "afl_gpu.cuh"
#include "macros.cuh"
#include <stdio.h>

#define WARP_SIZE 32
#define WORD_SIZE 32

__global__ void afl_compress_gpu (int bit_length, int *data, int *compressed_data, unsigned long length)
{
    unsigned int warp_th = (threadIdx.x % WARP_SIZE); 
    unsigned long pos = blockIdx.x * blockDim.x + threadIdx.x - warp_th;
    unsigned long data_id = pos * WARP_SIZE + warp_th;
    unsigned long cdata_id = pos * bit_length + warp_th;
    afl_compress_base_gpu(bit_length, data_id, cdata_id, data, compressed_data, length);
}

__global__ void afl_decompress_gpu (int bit_length, int *compressed_data, int * decompress_data, unsigned long length)
{
    unsigned int warp_th = (threadIdx.x % WARP_SIZE); 
    unsigned long pos = blockIdx.x * blockDim.x + threadIdx.x - warp_th;
    unsigned long data_id = pos * WARP_SIZE + warp_th;
    unsigned long cdata_id = pos * bit_length + warp_th;
    afl_decompress_base_gpu(bit_length, cdata_id, data_id, compressed_data, decompress_data, length);
}

__global__ void afl_decompress_value_gpu (int bit_length, int *compressed_data, int * decompress_data, unsigned long length)
{
    unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length)
    {
        decompress_data[tid] = afl_decompress_base_value_gpu(bit_length, compressed_data, tid);
    }
}

__host__ void run_afl_compress_gpu(int bit_length, int *data, int *compressed_data, unsigned long length)
{
    int block_size = WARP_SIZE * 8; // better occupancy 
    unsigned long block_number = (length + block_size * WARP_SIZE - 1) / (block_size * WARP_SIZE);
    afl_compress_gpu <<<block_number, block_size>>> (bit_length, data, compressed_data, length);
}

__host__ void run_afl_decompress_gpu(int bit_length, int *compressed_data, int *data, unsigned long length)
{
    int block_size = WARP_SIZE * 8; // better occupancy
    unsigned long block_number = (length + block_size * WARP_SIZE - 1) / (block_size * WARP_SIZE);
    afl_decompress_gpu <<<block_number, block_size>>> (bit_length, compressed_data, data, length);
}

__host__ void run_afl_decompress_value_gpu(int bit_length, int *compressed_data, int *data, unsigned long length)
{
    int block_size = WARP_SIZE * 8; // better occupancy
    unsigned long block_number = (length + block_size * WARP_SIZE - 1) / (block_size);
    afl_decompress_value_gpu <<<block_number, block_size>>> (bit_length, compressed_data, data, length);
}

__device__  void afl_compress_base_gpu (int bit_length, unsigned long data_id, unsigned long comp_data_id, int *data, int *compressed_data, unsigned long length)
{
    int v1, value = 0;
    unsigned int v1_pos=0, v1_len;
    unsigned long pos=comp_data_id, pos_data=data_id;

    for (unsigned int i = 0; i < WARP_SIZE && pos_data < length; ++i)
    {
        v1 = data[pos_data];
        pos_data += WARP_SIZE;

        if (v1_pos + bit_length >= WORD_SIZE){
            v1_len = WORD_SIZE - v1_pos;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);

            compressed_data[pos] = value;

            v1_pos = bit_length - v1_len;
            value = GETNPBITS(v1, v1_pos, v1_len); 

            pos += WARP_SIZE;  
        } else {
            v1_len = bit_length;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);
            v1_pos += v1_len;
        }
    }
    if (pos_data >= length  && pos_data < length + WARP_SIZE)
    {
        compressed_data[pos] = value;
    }
}

__device__ void afl_decompress_base_gpu (int bit_length, unsigned long comp_data_id, unsigned long data_id, int *compressed_data, int *data, unsigned long length)
{
    unsigned long pos = comp_data_id, pos_decomp = data_id;
    unsigned int v1_pos = 0, v1_len;
    int v1, ret;

    if (pos_decomp > length ) // Decompress not more elements then length
        return;
    v1 = compressed_data[pos];
    for (unsigned int i = 0; i < WARP_SIZE && pos_decomp < length; ++i)
    {
        if (v1_pos + bit_length >= WORD_SIZE){ 
            v1_len = WORD_SIZE - v1_pos;
            ret = GETNPBITS(v1, v1_len, v1_pos);

            pos += WARP_SIZE;  
            v1 = compressed_data[pos];

            v1_pos = bit_length - v1_len;
            ret = ret | ((GETNBITS(v1, v1_pos))<< v1_len);
        } else {
            v1_len = bit_length;
            ret = GETNPBITS(v1, v1_len, v1_pos);
            v1_pos += v1_len;
        }

        data[pos_decomp] = ret;
        pos_decomp += WARP_SIZE;
    }
}

__device__ int afl_decompress_base_value_gpu (
        int bit_length, 
        int *compressed_data, 
        unsigned long pos
        )
{
    int block = pos / (WARP_SIZE * WORD_SIZE);
    int pos_in_block = (pos % (WARP_SIZE * WORD_SIZE));
    int pos_in_warp_lane = pos_in_block % WARP_SIZE;
    int pos_in_warp_comp_block = pos_in_block / WARP_SIZE;

    int cblock_id = block * ( WARP_SIZE * bit_length)
        + pos_in_warp_lane 
        + ((pos_in_warp_comp_block * bit_length) / WORD_SIZE) * WARP_SIZE;

    int bit_pos = pos_in_warp_comp_block * bit_length % WORD_SIZE;
    int bit_ret = (WORD_SIZE - bit_pos) >= bit_length ? bit_length : WORD_SIZE - bit_pos;

    int ret = GETNPBITS(compressed_data[cblock_id], bit_ret, bit_pos);

    if (bit_ret < bit_length)
        ret |= GETNBITS(compressed_data[cblock_id+WARP_SIZE], bit_length - bit_ret) << bit_ret;

    return ret;
}

