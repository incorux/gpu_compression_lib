#ifndef TEST_DELTA_CUH_ISZ6QCRW
#define TEST_DELTA_CUH_ISZ6QCRW
#include "test_base.cuh"
#include "tools/data.cuh"
#include "compression/delta_gpu.cuh"


template <typename T, int CWARP_SIZE> 
class test_delta: public test_base<T, CWARP_SIZE> 
{
    public: 
        virtual void allocateMemory() {
            mmCudaMallocHost(this->manager,(void**)&this->host_data, this->dev_data_size_alloc);
            mmCudaMallocHost(this->manager,(void**)&this->host_data2, this->dev_data_size_alloc);

            // allocate maximal compressed data size rather than independent allocations for each compression ratio- improves testing time
            mmCudaMalloc(this->manager, (void **) &this->dev_out, this->dev_data_size_alloc); 
            mmCudaMalloc(this->manager, (void **) &this->dev_data, this->dev_data_size_alloc);

            mmCudaMalloc(this->manager, (void **) &this->dev_data_block_start, compression_blocks_count * sizeof(unsigned long));
        }

        virtual void setup(unsigned long max_size) {
            this->max_size = max_size;
            this->data_size = max_size * sizeof(T);
            this->cword = sizeof(T) * 8;

            // for size less then cword we actually will need more space than original data
            this->compressed_data_size = ((max_size < this->cword  ? this->cword : max_size) + 32) * sizeof(T);
            this->dev_data_size_alloc = this->compressed_data_size * sizeof(T);

            this->compression_blocks_count = (this->compressed_data_size / sizeof(T)) / CWARP_SIZE + 1;

        }

        virtual void initializeData(int bit_length) {
            big_random_block_with_decreasing_values ((unsigned long)this->max_size, bit_length, this->host_data);
        }

        // Clean up before compression
        virtual void cleanBeforeCompress() {
            cudaMemset(this->dev_out, 0, this->dev_data_size_alloc); 

            cudaMemset(this->dev_data_block_start, 0, compression_blocks_count * sizeof(unsigned long));
        }

        virtual void compressData(int bit_length) {
            run_delta_afl_compress_gpu <T, CWARP_SIZE> (bit_length, this->dev_data, this->dev_out, this->dev_data_block_start, this->max_size);
        }

        virtual void decompressData(int bit_length) {
            run_delta_afl_decompress_gpu <T, CWARP_SIZE> (bit_length, this->dev_out, this->dev_data_block_start, this->dev_data, this->max_size);
        }


    protected:
        T *dev_data_block_start;
        unsigned int compression_blocks_count;
        unsigned long dev_data_size_alloc;
};

#endif /* end of include guard: TEST_DELTA_CUH_ISZ6QCRW */
