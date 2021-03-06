#ifndef TEST_DELTA_PAFL_H
#define TEST_DELTA_PAFL_H
#include "test_pafl.cuh"
#include "test_delta.cuh"
#include "compression/pafl.cuh"

template <typename T, char CWARP_SIZE> 
class test_delta_pafl: public virtual test_pafl <T, CWARP_SIZE>, public virtual test_delta <T, CWARP_SIZE>
{
    virtual void allocateMemory() {
        test_base<T, CWARP_SIZE>::allocateMemory();
        test_pafl<T, CWARP_SIZE>::iner_allocateMemory();
        test_delta<T, CWARP_SIZE>::iner_allocateMemory();
    }

    virtual void setup(unsigned long max_size) {
        test_base<T, CWARP_SIZE>::setup(max_size);
        test_pafl<T, CWARP_SIZE> ::iner_setup(max_size);
        test_delta<T, CWARP_SIZE>::iner_setup(max_size);
    }
    virtual void initializeData(int bit_length) {
        if(bit_length > 30) bit_length = 30; //FIX

        big_random_block_with_decreasing_values_and_outliers ((unsigned long)this->max_size, bit_length, this->host_data, this->outlier_count);
    }

    // Clean up before compression
    virtual void cleanBeforeCompress() {
        test_base<T, CWARP_SIZE>::cleanBeforeCompress();
        test_pafl<T, CWARP_SIZE> :: iner_cleanBeforeCompress();
        test_delta<T, CWARP_SIZE>:: iner_cleanBeforeCompress();
    }

    virtual void compressData(int bit_length) {
            run_delta_pafl_compress_gpu_alternate <T,CWARP_SIZE> (
                bit_length,
                this->dev_data,
                this->dev_out,
                this->dev_data_block_start,
                this->max_size,

                this->dev_data_patch_values,
                this->dev_data_patch_index,
                this->dev_data_patch_count
                );

            cudaErrorCheck();
    }

    virtual void decompressData(int bit_length) {
            run_delta_pafl_decompress_gpu < T, CWARP_SIZE> (
                bit_length, 
                this->dev_out, 
                this->dev_data_block_start,
                this->dev_data, 
                this->max_size,

                this->dev_data_patch_values,
                this->dev_data_patch_index,
                this->dev_data_patch_count
                );

    }

    virtual void print_compressed_data_size(){
        //TODO: fix this
        unsigned long patch_count;
        unsigned long compression_blocks_count = test_delta<T, CWARP_SIZE>::compression_blocks_count;
        cudaMemcpy(&patch_count, this->dev_data_patch_count, sizeof(unsigned long), cudaMemcpyDeviceToHost);
        printf("Comp ratio %f",  (float)this->max_size / (patch_count * (sizeof(T) + sizeof(long)) +  compression_blocks_count * sizeof(T) + this->compressed_data_size));
        printf(" %d %lu %ld %ld %ld\n" , this->bit_length, this->max_size, this->data_size, this->compressed_data_size, patch_count);
    }
};

#endif /* TEST_DELTA_PAFL_H */

