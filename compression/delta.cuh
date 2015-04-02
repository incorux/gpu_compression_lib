#ifndef DELTA_CUH_KKZZHX97
#define DELTA_CUH_KKZZHX97


template < typename T>
__global__ void delta_compress(T *data, T* compressed_data, unsigned long length)
{
    unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;

    int laneId = threadIdx.x & 0x1f;
    int neighborId = (laneId - 1) ;

    int value1 = 0;
    int value2 = 0;

    for (int i = 0; i < CWORD_SIZE(T) && pos_data < length; ++i)
    {
        value1 = data[tid];
        value2 = __shfl_down(laneId, 1);

        compressData[tid] = value2;

        int result;

        if (laneId == 0)
            result
        
    }
}

void function_name(int pos, int *data, int *data_out)
{
    int x;
    int v = data[pos];
    int y=0; // tutaj otrzymamy x z thread obok
    if(threadIdx.x & (32-1) == 0)
        x = v;
    else
        x = v-y;
    data_out[pos]=x;

}

#endif /* end of include guard: DELTA_CUH_KKZZHX97 */
