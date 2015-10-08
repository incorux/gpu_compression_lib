#include "delta_gpu.cuh"
#include <stdio.h>

template <typename T>
__global__ void delta_compress_gpu (T *data, T *compressed_data, T *spoints, unsigned int bit_length, unsigned long length, unsigned long spoints_length) 
{
    int tid = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int laneId = threadIdx.x & 0x1f;
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
        value2 = __shfl( value1, neighborId, 32); // Get previous value from neighborId

        if (laneId == 0)
        {
            ret = zeroLaneValue - value1;
            zeroLaneValue = value2;
            printf("Thread %d final value = %d zeroLaneValue %d, value1 %d\n", threadIdx.x, zeroLaneValue - value1, zeroLaneValue, value1);
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


