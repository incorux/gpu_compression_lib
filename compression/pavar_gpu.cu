#include "pavar_gpu.cuh"
#include "avar_gpu.cuh"
#include "macros.cuh"

#include <stdio.h>

#define BQ_CAPACITY 2048

__global__ void pavar_compress_gpu (
        pavar_header comp_h, 
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

    pavar_compress_base_gpu(
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
    int pcount = private_patch_count[0] ;

    // reserve space for outliers
    if (threadIdx.x == 0) {
        if(pcount > BQ_CAPACITY) pcount = BQ_CAPACITY;
        old_global_patch_count[0] = atomicAdd(global_queue_patch_count, pcount);
    }

    __syncthreads();

    // write in paralel outliers to global memory
    int gpos = old_global_patch_count[0]; // get global memory index

    // Transfer data from shared to global, warning if pcount > BQ_CAPACITY transfer only BQ_CAPACITY 
    for (int i = threadIdx.x; i < pcount; i += blockDim.x) {
        global_queue_patch_values[gpos + i] = private_patch_values[i];
        global_queue_patch_index[gpos + i] = private_patch_index[i];
    }

    __syncthreads(); // Wait for all PATCHED values in global

    // Reuse current threads for PATCH compression
    int patch_count = global_queue_patch_count[0];

    /*printf("%d\n", patch_count);*/

    cdata_id = pos * comp_h.bit_length + warp_th; // reuse for PATCH compression
    
    //Compress values
    avar_header patch_header = {comp_h.patch_bit_length};
    avar_compress_base_gpu(patch_header, data_id, cdata_id, global_queue_patch_values, global_data_patch_values, patch_count);

    //Compress index
    patch_header.bit_length = 16; //TODO: policzyć względem rozmiaru danych
    avar_compress_base_gpu(patch_header, data_id, cdata_id, global_queue_patch_values, global_data_patch_index, patch_count);
}

__global__ void pavar_decompress_gpu (pavar_header comp_h, int *compressed_data, int * decompress_data, unsigned long length) //TODO: fix params list
{
    /*int warp_th = (threadIdx.x % WARP_SIZE);*/
    /*unsigned long pos = blockIdx.x * blockDim.x + threadIdx.x - warp_th;*/
    /*unsigned long data_id = pos * WARP_SIZE + warp_th;*/
    /*unsigned long cdata_id = pos * comp_h.bit_length + warp_th;*/

    /*avar_header cheader = {comp_h.patch_bit_length};*/
    /*avar_decompress_base_gpu(cheader, cdata_id, data_id, compressed_data, decompress_data, length);*/
}

__host__ void run_pavar_compress_gpu(
        pavar_header comp_h, 
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

    pavar_compress_gpu <<<block_number, block_size>>> (
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

__host__ void run_pavar_decompress_gpu(pavar_header comp_h, int *compressed_data, int *data, unsigned long length)
{
    /*int block_size = WARP_SIZE * 8; // better occupancy*/
    /*unsigned long block_number = (length + block_size * WARP_SIZE - 1) / (block_size * WARP_SIZE);*/
    /*avar_decompress_gpu <<<block_number, block_size>>> (comp_h, compressed_data, data, length);*/
    //TODO: nalozenie PATCH
}

__device__  void pavar_compress_base_gpu (
        pavar_header comp_h, 

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
    unsigned int mask = NBITSTOMASK(comp_h.patch_bit_length);

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
                private_patch_values[p_pos] = v1;
                private_patch_index[p_pos] = pos_decomp;
            } else {
                // in case shared memory is full
                p_pos = atomicAdd(global_patch_count, 1);

                global_patch_values[p_pos] = v1;
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
