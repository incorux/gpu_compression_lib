#ifndef DELTA_CUH_KKZZHX97
#define DELTA_CUH_KKZZHX97


template <typename T>
__global__ void delta_compress_gpu (T *data, T *compressed_data, unsigned long length) 
{
    int tid = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int laneId = threadIdx.x & 0x1f;

    int value1=0;
    int value2=0;
    int zeroLaneValue=0;
    char neighborId = laneId - 1;

    //TODO: add if data index in range
    value1 = ( 32 * 32 ) - laneId; //TODO: Read from data store
    zeroLaneValue = value1;

    if (laneId == 0)  neighborId = 31; 

    int ret = 0;

    for (int i = 1;  i < 32; ++ i)
    {
        value2 = __shfl( value1, neighborId, 32); // Get previous value from neighborId

        if (laneId == 0)
        {
            printf("Thread %d final value = %d zeroLaneValue %d, value1 %d\n", threadIdx.x, zeroLaneValue - value1, zeroLaneValue, value1);
            ret = zeroLaneValue - value1;
            zeroLaneValue = value2;
        } else {
            ret = value2 - value1;
            printf("Thread %d final value = %d\n", threadIdx.x, value2 - value1);
        }

        value1 = (32*32) - laneId - i * 32; //TODO: Read next value from data store

        if (ret != 1)
        {
            printf("Error %d ret %d\n", threadIdx.x, ret);
        }
    }
}


template <typename T>
__global__ void delta_decompress_gpu (T *compressed_data, T *data, unsigned long length, int width=32)
{
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int lane_id = id % warpSize;
    int value = data[id];
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

#endif /* end of include guard: DELTA_CUH_KKZZHX97 */
