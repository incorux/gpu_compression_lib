#include "delta_gpu.cuh"
#include <stdio.h>

#define WARP_SZ 32
__device__ inline int get_lane_id(void) { return threadIdx.x % WARP_SZ; }

__device__ int shfl_prefix_sum(int value, int width=32) 
{
    int lane_id = get_lane_id();

    // Now accumulate in log2(32) steps
#pragma unroll
    for(int i=1; i<=width; i*=2) {
        int n = __shfl_up(value, i);
        if(lane_id >= i) value += n;
    }

    return value;
}

__device__ long shfl_prefix_sum(long value, int width=32) 
{
//TODO obsluga dla long trzeba wykonac podwojna zmiane
    int lane_id = get_lane_id();

    // Now accumulate in log2(32) steps
#pragma unroll
    for(int i=1; i<=width; i*=2) {
        int n = __shfl_up(value, i);
        if(lane_id >= i) value += n;
    }

    return value;
}

template <typename T, char CWARP_SIZE>
__device__  void delta_afl_compress_base_gpu (const unsigned int bit_length, unsigned long data_id, unsigned long comp_data_id, T *data, T *compressed_data, T* compressed_data_block_start, unsigned long length)
{

    T v1, value = 0;
    unsigned int v1_pos=0, v1_len;
    unsigned long pos=comp_data_id, pos_data=data_id;

    T zeroLaneValue, v2;
    const unsigned long lane = get_lane_id();
    char neighborId = lane - 1;

    const unsigned int warp_lane = (threadIdx.x % CWARP_SIZE); 
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;

    if (lane == 0)  {
        neighborId = 31; 
        zeroLaneValue = data[pos_data];
        compressed_data_block_start[data_block] = zeroLaneValue;
    }

    for (unsigned int i = 0; i < CWORD_SIZE(T) && pos_data < length; ++i) 
    {
        v1 = data[pos_data];
        pos_data += CWARP_SIZE;
        
        //TODO: v1 reduction 
        v2 = __shfl( v1, neighborId, 32); 

        if (lane == 0)
        {
            // Lane 0 uses data from previous iteration
            v1 = zeroLaneValue - v1; 
            zeroLaneValue = v2;
        } else {
            v1 = v2 - v1;
        }

        if (v1_pos >= CWORD_SIZE(T) - bit_length){
            v1_len = CWORD_SIZE(T) - v1_pos;
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

template <typename T, char CWARP_SIZE>
__device__ void delta_afl_decompress_base_gpu (
        const unsigned int bit_length, 
        unsigned long comp_data_id,
        unsigned long data_id, 
        T *compressed_data, 
        T* compressed_data_block_start, 
        T *data, 
        unsigned long length
        )
{
    unsigned long pos = comp_data_id, pos_decomp = data_id;
    unsigned int v1_pos = 0, v1_len;
    T v1, ret;

    const unsigned long lane = get_lane_id();

    if (pos_decomp > length ) // Decompress not more elements then length
        return;
    v1 = compressed_data[pos];

    T zeroLaneValue, v2;

    const unsigned int warp_lane = (threadIdx.x % CWARP_SIZE); 
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;

    if (lane == 0) {
       zeroLaneValue = compressed_data_block_start[data_block];
    }

    for (unsigned int i = 0; i < CWORD_SIZE(T) && pos_decomp < length; ++i)
    {
        if (v1_pos >= CWORD_SIZE(T) - bit_length){ 
            v1_len = CWORD_SIZE(T) - v1_pos;
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

        v2 = __shfl(zeroLaneValue, 0); // get zero, lane value from lane 0 
        ret += v2;

        data[pos_decomp] = ret;
        pos_decomp += CWARP_SIZE;

        v2 = __shfl(ret, 31); // get final ret from lane 31
        if(lane == 0) 
            zeroLaneValue = v2; 
    }
}

template <typename T>
__global__ void delta_compress_gpu (T *data, T *compressed_data, T *spoints, unsigned int bit_length, unsigned long length, unsigned long spoints_length) 
{
    int tid = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int laneId = get_lane_id();
    int warpId = tid / 32; //TODO: fix

    int value1=0;
    int value2=0;
    int zeroLaneValue=0;

    char neighborId = laneId - 1;

    //TODO: add if data index in range
    value1 = laneId; //data[tid];
    zeroLaneValue = value1;

    if (laneId == 0)  {
        neighborId = 31; 
        spoints[warpId] = value1;
    }

    int ret = 0;

    for (int i = 1;  i < 2; ++ i)
    {
        // Get previous value from neighborId
        value2 = __shfl( value1, neighborId, 32); 

        if (laneId == 0)
        {
            // Lane 0 operates wraps data for next iteration
            ret = zeroLaneValue - value1;
            zeroLaneValue = value2;
            
            /* printf("Thread %d final value = %d zeroLaneValue %d, value1 %d\n", threadIdx.x, zeroLaneValue - value1, zeroLaneValue, value1); */
        } else {
            ret = value2 - value1;
            printf("Thread %d final value = %d\n", threadIdx.x, value2 - value1);
        }

        value1 = laneId + i; //data[tid + i * 32]; 

        /* if (ret != 1) */
        /* { */
        /*     printf("Error %d ret %d\n", threadIdx.x, ret); */
        /* } */
    }
}

template <typename T>
__global__ void delta_decompress_gpu (T *compressed_data, T *spoints, T *data, unsigned long length, unsigned int spoints_length, int width=32)
{
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int lane_id = id % warpSize;
    int warpId = threadIdx.x / 32; //TODO: fix

    T value = data[id] + spoints[warpId];
    int laneZeroValue=0;

    for (int j = 0;  j < 3; ++ j)
    {
        // Now accumulate in log2(32) steps
        for(int i=1; i <= width; i *= 2) 
        {
            int n = __shfl_up(value, i);
            if(lane_id >= i) value += n;
        }
        
        printf("Thread %d final value = %d\n", threadIdx.x, value);
        laneZeroValue = __shfl(value, 31);
        value = data[id];

        if (lane_id == 0) value=laneZeroValue + value;
    }
}

template __global__ void delta_compress_gpu <int> (int *data, int *compressed_data, int *spoints, unsigned int bit_length, unsigned long length, unsigned long spoints_length);

