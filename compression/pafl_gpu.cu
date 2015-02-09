#include "pafl_gpu.cuh"
#include "afl_gpu.cuh"
#include "macros.cuh"
#include "tools.cuh"
#include <math.h>

#include <stdio.h>

//TODO: przygotowac kod który po przepełnienu shared będzie transferował dane do global

#define BQ_CAPACITY 1800 // TODO: FINE TUNE
#define BQ_CAPACITY_ALTERNATE 512 // TODO: FINE TUNE

__global__ void pafl_compress_gpu_alternate2 (
        pafl_header comp_h, 
        int *data, 
        int *compressed_data, 
        unsigned long length,

        int *global_queue_patch_values,
        int *global_queue_patch_index,
        int *global_queue_patch_count
        )
{
    unsigned long tid =  blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int private_warp_patch_values[32][16];
    __shared__ int private_warp_patch_index [32][16];

    __shared__ int private_warp_patch_count[32];

    __shared__ int private_block_patch_values[32*16];
    __shared__ int private_block_patch_index [32*16];
    __shared__ int private_block_patch_count [1];
    __shared__ int old_global_patch_count[1];

    unsigned int mask = ~NBITSTOMASK(comp_h.bit_length + comp_h.patch_bit_length);

    if (threadIdx.x == 0) 
        private_block_patch_count[0] = 0;

    if (threadIdx.x < 32) // Warp_queue to block_queue
        private_warp_patch_count[threadIdx.x] = 0;

    __syncthreads(); // Gather outliers into warp_queue
    if (tid < length) //PATCH
    {
        int v1 = data[tid];
        if ( v1 & mask) {
            int p = private_warp_patch_count[threadIdx.x % 32]++; //TODO: czy wystarczy tak czy atomic
            private_warp_patch_index[threadIdx.x%32][p] = tid;
            private_warp_patch_values[threadIdx.x%32][p] = v1;
        }
    }

    __syncthreads(); // Warp_queue to block_queue
    if (threadIdx.x < 32) 
    { 
        int c = private_warp_patch_count[threadIdx.x];
        int pos = atomicAdd(private_block_patch_count, c);
        for (int i = 0; i < c; ++i)
        {
            private_block_patch_values[pos+i] = private_warp_patch_values[threadIdx.x][i];
            private_block_patch_index[pos+i] = private_warp_patch_index[threadIdx.x][i];
        }
    }

    __syncthreads(); // Block queue to global queue

    int pcount = private_block_patch_count[0];

    if (threadIdx.x == 0) {
        old_global_patch_count[0] = atomicAdd(global_queue_patch_count, pcount);
    }

    __syncthreads();

    int gpos = old_global_patch_count[0]; // get global memory index

    //TODO: Transfer compressed to global ??
    // Transfer data from shared to global, write in paralel outliers to global memory
    if (threadIdx.x < pcount) {
        global_queue_patch_values[gpos + threadIdx.x] = private_block_patch_values[threadIdx.x];
        global_queue_patch_index[gpos + threadIdx.x] = private_block_patch_index[threadIdx.x];
    }

}

__global__ void pafl_compress_gpu_alternate (
        pafl_header comp_h, 
        int *data, 
        int *compressed_data, 
        unsigned long length,

        int *global_queue_patch_values,
        int *global_queue_patch_index,
        int *global_queue_patch_count
        )
{
    unsigned long tid =  blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int private_block_patch_values[32*16];
    __shared__ int private_block_patch_index [32*16];
    __shared__ int private_block_patch_count [1];
    __shared__ int old_global_patch_count[1];

    /*unsigned int mask = NBITSTOMASK(comp_h.bit_length + comp_h.patch_bit_length);*/
    unsigned int mask = ~NBITSTOMASK(comp_h.bit_length);

    if (threadIdx.x == 0) 
        private_block_patch_count[0] = 0;

    __syncthreads(); // Gather outliers into warp_queue
    if (tid < length) //PATCH
    {
        int v1 = data[tid];
        if ( v1 & mask) {
            int p = atomicAdd(private_block_patch_count,1);
            private_block_patch_index[p] = (int) tid; //TODO: fis later
            private_block_patch_values[p] = (v1 >> comp_h.bit_length);//GETNPBITS(v1, comp_h.patch_bit_length, comp_h.bit_length);
            /*printf(">> tid = %ld, %d, v1 = %d p=%d b=%d rv1=%d p_b_l=%d<<\n", tid, GETNPBITS(private_block_patch_index[p], BITLEN(length) , data[tid], private_block_patch_values[p], GETNBITS(v1, comp_h.bit_length), GETNBITS(v1, comp_h.bit_length) | (private_block_patch_values[p] << comp_h.bit_length), comp_h.bit_length);*/
        }
    }

    __syncthreads(); // Block queue to global queue

    int pcount = private_block_patch_count[0];

    if (threadIdx.x == 0) {
        old_global_patch_count[0] = atomicAdd(global_queue_patch_count, pcount);
    }

    __syncthreads();

    int gpos = old_global_patch_count[0]; // get global memory index

    // Transfer data from shared to global, write in paralel outliers to global memory
    if (threadIdx.x < pcount) {
        global_queue_patch_values[gpos + threadIdx.x] = private_block_patch_values[threadIdx.x];
        global_queue_patch_index[gpos + threadIdx.x] = private_block_patch_index[threadIdx.x];
        /*printf("IDX %d %d", private_block_patch_index[threadIdx.x], private_block_patch_values[threadIdx.x]);*/
    }
}

__global__ void pafl_compress_gpu (
        pafl_header comp_h, 
        int *data, 
        int *compressed_data, 
        unsigned long length,

        int *global_queue_patch_values,
        int *global_queue_patch_index,
        int *global_queue_patch_count,

        int *global_data_patch_values,
        int *global_data_patch_index,
        int *global_data_patch_count
        )
{
    int warp_th = (threadIdx.x % WARP_SIZE); 
    unsigned long pos = blockIdx.x * blockDim.x + threadIdx.x - warp_th;
    unsigned long data_id = pos * WARP_SIZE + warp_th;
    unsigned long cdata_id = pos * comp_h.bit_length + warp_th;

    __shared__ int private_patch_values[BQ_CAPACITY];
    __shared__ int private_patch_index[BQ_CAPACITY];

    __shared__ int private_patch_count[1];
    __shared__ int old_global_patch_count[1];

    if(threadIdx.x == 0) 
        private_patch_count[0] = 0;

    __syncthreads();

    pafl_compress_base_gpu(
            comp_h, 
            data_id, 
            cdata_id, 

            data, 
            compressed_data, 
            length,

            private_patch_values,
            private_patch_index,
            private_patch_count,

            global_queue_patch_values,
            global_queue_patch_index,
            global_queue_patch_count
            );

    __syncthreads(); // Wait for all PATCHED values to be stored in shared

    // get outliers count from shared memory
    int pcount = private_patch_count[0] > BQ_CAPACITY ? BQ_CAPACITY : private_patch_count[0]; //warning if pcount > BQ_CAPACITY transfer only BQ_CAPACITY 

    // reserve space for outliers
    if (threadIdx.x == 0) {
        old_global_patch_count[0] = atomicAdd(global_queue_patch_count, pcount);
        /*printf("b %d c %d\n", blockIdx.x, private_patch_count[0] );*/
    }

    __syncthreads();

    int gpos = old_global_patch_count[0]; // get global memory index

    // Transfer data from shared to global, write in paralel outliers to global memory
    for (int i = threadIdx.x; i < pcount; i += blockDim.x) {
        global_queue_patch_values[gpos + i] = private_patch_values[i];
        global_queue_patch_index[gpos + i] = private_patch_index[i];
    }

    __syncthreads(); // Wait for all PATCHED values in global


    //TODO: tak raczej nie mozna ! nie mamy gwarancji ze wszystkie bloki skoncza dzialac w tym samym momencie 

    // Reuse current threads for PATCH compression
    int patch_count = global_queue_patch_count[0];
    
    int patch_values_bit_length = comp_h.patch_bit_length;
    //Compress values
    cdata_id = pos * patch_values_bit_length + warp_th; // reuse for PATCH compression
    afl_compress_base_gpu<int, 32, 32>(patch_values_bit_length, data_id, cdata_id, global_queue_patch_values, global_data_patch_values, patch_count);

    int patch_index_bit_length = BITLEN(length);
    //Compress index
    cdata_id = pos * patch_index_bit_length + warp_th; // reuse for PATCH compression
    afl_compress_base_gpu<int, 32, 32>(patch_index_bit_length, data_id, cdata_id, global_queue_patch_values, global_data_patch_index, patch_count);
}

__global__ void patch_apply_gpu (
        pafl_header comp_h,
        int *decompressed_data,
        unsigned long length,
        
        int *global_data_patch_values,
        int *global_data_patch_index,
        int *global_data_patch_count
        ) //TODO: fix params list
{
    unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;

    int patch_length = *global_data_patch_count;

    //TODO: Run decompression on selected starting threads
    if (tid < patch_length)
    {
        int idx = afl_decompress_base_value_gpu<int, 32, 32>((int)log2((float)length)+1, global_data_patch_index, tid);
        //TODO: tu chyba jest blad

        int val = afl_decompress_base_value_gpu<int, 32, 32>(comp_h.patch_bit_length, global_data_patch_values, tid);
        /*printf("DIDX %d\n", idx);*/

        decompressed_data[idx] |= (val << comp_h.bit_length); //TODO: check if idx <length ??
        /*printf("DIDX %d %d %d %d\n", idx, decompressed_data[idx], (val << comp_h.bit_length), decompressed_data[idx] | (val << comp_h.bit_length));*/

        /*printf("DIDX %d %d\n", idx, decompressed_data[idx]);*/
    }
}

__host__ void run_pafl_compress_gpu_alternate(
        pafl_header comp_h, 
        int *data, 
        int *compressed_data, 
        unsigned long length,
        
        int *global_queue_patch_values,
        int *global_queue_patch_index,
        int *global_queue_patch_count,

        int *global_data_patch_values,
        int *global_data_patch_index,
        int *global_data_patch_count
        )
{
    int block_size = 512; // better occupancy
    unsigned long block_number = (length + block_size * WARP_SIZE - 1) / (block_size);

    pafl_compress_gpu_alternate <<<block_number, block_size>>> (
            comp_h, 
            data, 
            compressed_data, 
            length,

            global_queue_patch_values,
            global_queue_patch_index,
            global_queue_patch_count
            );

    block_size = WARP_SIZE * 8; // better occupancy 
    block_number = (length + block_size * WARP_SIZE - 1) / (block_size * WARP_SIZE);

    afl_compress_gpu <int, 32, 32> <<<block_number, block_size>>> (comp_h.bit_length, data, compressed_data, length);

    //Patch compress
    int patch_count;
    gpuErrchk(cudaMemcpy(&patch_count, global_queue_patch_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (patch_count > 0)
    {
        block_size = WARP_SIZE * 8; // better occupancy 
        block_number = (patch_count + block_size * WARP_SIZE - 1) / (block_size * WARP_SIZE);

        afl_compress_gpu <int, 32, 32><<<block_number, block_size>>> (comp_h.patch_bit_length, global_queue_patch_values, global_data_patch_values, patch_count);

        afl_compress_gpu <int, 32, 32> <<<block_number, block_size>>> ((int)log2((float)length)+1, global_queue_patch_index, global_data_patch_index, patch_count);
    }
}

__host__ void run_pafl_compress_gpu(
        pafl_header comp_h, 
        int *data, 
        int *compressed_data, 
        unsigned long length,
        
        int *global_queue_patch_values,
        int *global_queue_patch_index,
        int *global_queue_patch_count,

        int *global_data_patch_values,
        int *global_data_patch_index,
        int *global_data_patch_count
        )
{
    int block_size = WARP_SIZE * 8; // better occupancy
    unsigned long block_number = (length + block_size * WARP_SIZE - 1) / (block_size * WARP_SIZE);

    pafl_compress_gpu <<<block_number, block_size>>> (
            comp_h, 
            data, 
            compressed_data, 
            length,

            global_queue_patch_values,
            global_queue_patch_index,
            global_queue_patch_count,

            global_data_patch_values,
            global_data_patch_index,
            global_data_patch_count
            );
}

__host__ void run_pafl_decompress_gpu(
        pafl_header comp_h, 
        int *compressed_data, 
        int *data, 
        unsigned long length,
        
        int *global_data_patch_values,
        int *global_data_patch_index,
        int *global_data_patch_count
        )
{
    int block_size = WARP_SIZE * 8; // better occupancy
    unsigned long block_number = (length + block_size * WARP_SIZE - 1) / (block_size * WARP_SIZE);
    afl_decompress_gpu <int, 32, 32> <<<block_number, block_size>>> (comp_h.bit_length, compressed_data, data, length);

    /*cudaErrorCheck();*/
    patch_apply_gpu <<<block_number * WARP_SIZE, block_size>>> (
            comp_h, 
            data, 
            length,

            global_data_patch_values,
            global_data_patch_index,
            global_data_patch_count
            );
}

__device__  void pafl_compress_base_gpu (
        pafl_header comp_h, 

        unsigned long data_id, 
        unsigned long comp_data_id, 

        int *data, 
        int *compressed_data, 
        unsigned long length,

        int *private_patch_values,
        int *private_patch_index,
        int *private_patch_count,

        int *global_patch_values,
        int *global_patch_index,
        int *global_patch_count
        )
{
    int v1, value = 0;
    unsigned int v1_pos=0, v1_len;
    unsigned long pos=comp_data_id, pos_decomp=data_id;
    unsigned int mask = ~NBITSTOMASK(comp_h.bit_length + comp_h.patch_bit_length);

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

        // check if value is outlier, if so write it to shared memory (or global if shared is full)
        if ( v1 & mask) {
            unsigned int p_pos = atomicAdd(private_patch_count, 1);
            if (p_pos < BQ_CAPACITY) {
                private_patch_values[p_pos] = v1 >> comp_h.bit_length;
                private_patch_index[p_pos] = pos_decomp;
            } else {
                // in case shared memory is full
                p_pos = atomicAdd(global_patch_count, 1);

                global_patch_values[p_pos] = v1 >> comp_h.bit_length;
                global_patch_index[p_pos] = pos_decomp;
            }
        }
    }

    if (pos_decomp >= length  && pos_decomp < length + WARP_SIZE)
        compressed_data[pos] = value;
}


/*
TODO: mozna to jeszcze bardziej poprawic (tzn. dokonywac kompresji w locie poprzez atomowe operacje bitowe
tzn. kazdy z nich dostaje id elementu, wylicza ktorych elementow to dotyczy i robi na nich atomowe OR
Wazne pamietac o wyzerowaniu shared memory na poczatku
*/
/*
TODO:
  * kompresujemy dane w shared i zapisujemy do global (wyrownojemy kompresje - tzn. nadmiarowe elementy zapisujemy do global)
  * na koncu kompresuj dane z global (threadfence albo nowe wywolanie)
  */
