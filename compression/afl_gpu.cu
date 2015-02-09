#include "afl_gpu.cuh"
#include "macros.cuh"
#include <stdio.h>

template < typename T, char CWORD_SIZE, char CWARP_SIZE >
__host__ void run_afl_compress_gpu(int bit_length, T *data, T *compressed_data, unsigned long length)
{
    int block_size = CWARP_SIZE * 8; // better occupancy 
    unsigned long block_number = (length + block_size * CWARP_SIZE - 1) / (block_size * CWARP_SIZE);
    afl_compress_gpu <T, CWORD_SIZE, CWARP_SIZE> <<<block_number, block_size>>> (bit_length, data, compressed_data, length);
}

template < typename T, char CWORD_SIZE, char CWARP_SIZE >
__host__ void run_afl_decompress_gpu(int bit_length, T *compressed_data, T *data, unsigned long length)
{
    int block_size = CWARP_SIZE * 8; // better occupancy
    unsigned long block_number = (length + block_size * CWARP_SIZE - 1) / (block_size * CWARP_SIZE);
    afl_decompress_gpu <T, CWORD_SIZE, CWARP_SIZE> <<<block_number, block_size>>> (bit_length, compressed_data, data, length);
}

template < typename T, char CWORD_SIZE, char CWARP_SIZE >
__host__ void run_afl_decompress_value_gpu(int bit_length, T *compressed_data, T *data, unsigned long length)
{
    int block_size = CWARP_SIZE * 8; // better occupancy
    unsigned long block_number = (length + block_size * CWARP_SIZE - 1) / (block_size);
    afl_decompress_value_gpu <T, CWORD_SIZE, CWARP_SIZE> <<<block_number, block_size>>> (bit_length, compressed_data, data, length);
}

template < typename T, char CWORD_SIZE, char CWARP_SIZE >
__global__ void afl_compress_gpu (int bit_length, T *data, T *compressed_data, unsigned long length)
{
    unsigned int warp_th = (threadIdx.x % CWARP_SIZE); 
    unsigned long pos = blockIdx.x * blockDim.x + threadIdx.x - warp_th;
    unsigned long data_id = pos * CWARP_SIZE + warp_th;
    unsigned long cdata_id = pos * bit_length + warp_th;

    afl_compress_base_gpu <T, CWORD_SIZE, CWARP_SIZE> (bit_length, data_id, cdata_id, data, compressed_data, length);
}

template < typename T, char CWORD_SIZE, char CWARP_SIZE >
__global__ void afl_decompress_gpu (int bit_length, T *compressed_data, T * decompress_data, unsigned long length)
{
    unsigned int warp_th = (threadIdx.x % CWARP_SIZE); 
    unsigned long pos = blockIdx.x * blockDim.x + threadIdx.x - warp_th;
    unsigned long data_id = pos * CWARP_SIZE + warp_th;
    unsigned long cdata_id = pos * bit_length + warp_th;

    afl_decompress_base_gpu <T, CWORD_SIZE, CWARP_SIZE> (bit_length, cdata_id, data_id, compressed_data, decompress_data, length);
}

template < typename T, char CWORD_SIZE, char CWARP_SIZE >
__global__ void afl_decompress_value_gpu (int bit_length, T *compressed_data, T * decompress_data, unsigned long length)
{
    unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length)
    {
        decompress_data[tid] = afl_decompress_base_value_gpu <T, CWORD_SIZE, CWARP_SIZE> (bit_length, compressed_data, tid);
    }
}


template <typename T, char CWORD_SIZE, char CWARP_SIZE>
__device__  __host__ void afl_compress_base_gpu (int bit_length, unsigned long data_id, unsigned long comp_data_id, T *data, T *compressed_data, unsigned long length)
{
    int v1, value = 0;
    unsigned int v1_pos=0, v1_len;
    unsigned long pos=comp_data_id, pos_data=data_id;

    for (unsigned int i = 0; i < CWARP_SIZE && pos_data < length; ++i)
    {
        v1 = data[pos_data];
        pos_data += CWARP_SIZE;

        if (v1_pos + bit_length >= CWORD_SIZE){
            v1_len = CWORD_SIZE - v1_pos;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);

            compressed_data[pos] = value;

            v1_pos = bit_length - v1_len;
            value = GETNPBITS(v1, v1_pos, v1_len); 

            pos += CWARP_SIZE;  
        } else {
            v1_len = bit_length;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);
            v1_pos += v1_len;
        }
    }
    if (pos_data >= length  && pos_data < length + CWARP_SIZE)
    {
        compressed_data[pos] = value;
    }
}

template <typename T, char CWORD_SIZE, char CWARP_SIZE>
__device__ __host__ void afl_decompress_base_gpu (int bit_length, unsigned long comp_data_id, unsigned long data_id, T *compressed_data, T *data, unsigned long length)
{
    unsigned long pos = comp_data_id, pos_decomp = data_id;
    unsigned int v1_pos = 0, v1_len;
    int v1, ret;

    if (pos_decomp > length ) // Decompress not more elements then length
        return;
    v1 = compressed_data[pos];
    for (unsigned int i = 0; i < CWARP_SIZE && pos_decomp < length; ++i)
    {
        if (v1_pos + bit_length >= CWORD_SIZE){ 
            v1_len = CWORD_SIZE - v1_pos;
            ret = GETNPBITS(v1, v1_len, v1_pos);

            pos += CWARP_SIZE;  
            v1 = compressed_data[pos];

            v1_pos = bit_length - v1_len;
            ret = ret | ((GETNBITS(v1, v1_pos))<< v1_len);
        } else {
            v1_len = bit_length;
            ret = GETNPBITS(v1, v1_len, v1_pos);
            v1_pos += v1_len;
        }

        data[pos_decomp] = ret;
        pos_decomp += CWARP_SIZE;
    }
}

template <typename T, char CWORD_SIZE, char CWARP_SIZE>
__device__ __host__ T afl_decompress_base_value_gpu (
        int bit_length, 
        T *compressed_data, 
        unsigned long pos
        )
{
    int block = pos / (CWARP_SIZE * CWORD_SIZE);
    int pos_in_block = (pos % (CWARP_SIZE * CWORD_SIZE));
    int pos_in_warp_lane = pos_in_block % CWARP_SIZE;
    int pos_in_warp_comp_block = pos_in_block / CWARP_SIZE;

    unsigned long cblock_id = block * ( CWARP_SIZE * bit_length)
        + pos_in_warp_lane 
        + ((pos_in_warp_comp_block * bit_length) / CWORD_SIZE) * CWARP_SIZE;

    int bit_pos = pos_in_warp_comp_block * bit_length % CWORD_SIZE;
    int bit_ret = (CWORD_SIZE - bit_pos) >= bit_length ? bit_length : CWORD_SIZE - bit_pos;

    int ret = GETNPBITS(compressed_data[cblock_id], bit_ret, bit_pos);

    if (bit_ret < bit_length)
        ret |= GETNBITS(compressed_data[cblock_id+CWARP_SIZE], bit_length - bit_ret) << bit_ret;

    return ret;
}

template __device__ __host__ int afl_decompress_base_value_gpu <int, 32, 32> (int, int*, unsigned long);
template __device__ __host__ long afl_decompress_base_value_gpu <long, 32, 32>(int, long*, unsigned long);

template __device__ __host__ void afl_decompress_base_gpu <int, 32, 32> (int, unsigned long comp_data_id, unsigned long data_id, int *compressed_data, int *data, unsigned long length);
template __device__ __host__ void afl_decompress_base_gpu <long, 32, 32> (int, unsigned long comp_data_id, unsigned long data_id, long *compressed_data, long *data, unsigned long length);

template __device__  __host__ void afl_compress_base_gpu <int, 32, 32> (int, unsigned long, unsigned long, int *, int *, unsigned long );
template __device__  __host__ void afl_compress_base_gpu <long, 32, 32> (int, unsigned long, unsigned long, long *, long *, unsigned long );


template __global__ void afl_decompress_value_gpu < int, 32, 32> ( int bit_length, int *compressed_data , int * decompress_data , unsigned long length);
template __global__ void afl_compress_gpu         < int, 32, 32> ( int bit_length, int *data, int *compressed_data, unsigned long length);
template __global__ void afl_decompress_gpu       < int, 32, 32> ( int bit_length, int *compressed_data, int * decompress_data, unsigned long length);

template __global__ void afl_decompress_value_gpu < long, 32, 32> ( int bit_length, long *compressed_data , long * decompress_data , unsigned long length);
template __global__ void afl_compress_gpu         < long, 32, 32> ( int bit_length, long *data, long *compressed_data, unsigned long length);
template __global__ void afl_decompress_gpu       < long, 32, 32> ( int bit_length, long *compressed_data, long * decompress_data, unsigned long length);


template __host__ void run_afl_compress_gpu         < int, 32, 32> (int bit_length, int *data, int *compressed_data, unsigned long length);
template __host__ void run_afl_decompress_value_gpu < int, 32, 32> (int bit_length, int *compressed_data , int *data, unsigned long length);
template __host__ void run_afl_decompress_gpu       < int, 32, 32> (int bit_length, int *data, int *compressed_data, unsigned long length);

template __host__ void run_afl_compress_gpu         < long, 32, 32> (int bit_length, long *data, long *compressed_data, unsigned long length);
template __host__ void run_afl_decompress_value_gpu < long, 32, 32> (int bit_length, long *compressed_data , long *data, unsigned long length);
template __host__ void run_afl_decompress_gpu       < long, 32, 32> (int bit_length, long *data, long *compressed_data, unsigned long length);
