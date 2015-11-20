#ifndef TEST_AAFL_CUH_BUYDHFII
#define TEST_AAFL_CUH_BUYDHFII

#include "test_base.cuh"
#include "compression/aafl_gpu.cuh"

template <typename T, int CWARP_SIZE> 
class test_aafl: public test_base<T, CWARP_SIZE> 
{
    public: 
        virtual void allocateMemory() {

            test_base<T, CWARP_SIZE>::allocateMemory();
            /* mmCudaMallocHost(this->manager,(void**)&this->host_data, this->compressed_data_size); */
            /* mmCudaMallocHost(this->manager,(void**)&this->host_data2, this->compressed_data_size); */

            /* // allocate maximal compressed data size rather than independent allocations for each compression ratio- improves testing time */
            /* mmCudaMalloc(this->manager, (void **) &this->dev_out, this->compressed_data_size); */ 
            /* mmCudaMalloc(this->manager, (void **) &this->dev_data, this->compressed_data_size); */

            mmCudaMalloc(this->manager, (void **) &this->dev_data_compressed_data_register, sizeof(long));
            mmCudaMalloc(this->manager, (void **) &this->dev_data_bit_lenght, compression_blocks_count * sizeof(unsigned char));
            mmCudaMalloc(this->manager, (void **) &this->dev_data_position_id, compression_blocks_count * sizeof(unsigned long));
        }

        virtual void setup(unsigned long max_size) {
            test_base<T, CWARP_SIZE>::setup(max_size);

            unsigned long data_block_size = CWORD_SIZE(T) * 8;

            /* // for size less then cword we actually will need more space than original data */
            /* this->compressed_data_size = max_size < data_block_size ? data_block_size : max_size; */

            /* // TODO: added * 32 to allocation space, without there is some under allocation - to be checked !! */
            /* this->compressed_data_size = ((this->compressed_data_size + data_block_size - 1) / data_block_size) * data_block_size * sizeof(T) + 4048*sizeof(T); */

            this->compression_blocks_count = this->max_size / data_block_size  + 1024;

            this->compressed_data_size += 2048 * sizeof(T);
        }

        virtual void initializeData(int bit_length) {
            big_random_block(this->max_size, bit_length, this->host_data);
        }

        // Clean up before compression
        virtual void cleanBeforeCompress() {
            test_base<T, CWARP_SIZE>::cleanBeforeCompress();

            cudaMemset(this->dev_data_compressed_data_register, 0, sizeof(long)); 
            cudaMemset(this->dev_data_bit_lenght, 0, compression_blocks_count * sizeof(unsigned char));
            cudaMemset(this->dev_data_position_id, 0, compression_blocks_count * sizeof(unsigned long));
        }

        virtual void compressData(int bit_length) {
            run_aafl_compress_gpu <T, CWARP_SIZE> (this->dev_data_compressed_data_register, this->dev_data_bit_lenght, this->dev_data_position_id, this->dev_data, this->dev_out, this->max_size);
        }

        virtual void decompressData(int bit_length) {
            run_aafl_decompress_gpu <T, CWARP_SIZE> (this->dev_data_bit_lenght, this->dev_data_position_id, this->dev_out, this->dev_data, this->max_size);
        }


    protected:
        unsigned char *dev_data_bit_lenght;
        unsigned long *dev_data_position_id;
        unsigned long compression_blocks_count;
        unsigned long *dev_data_compressed_data_register;
};

#endif /* end of include guard: TEST_AAFL_CUH_BUYDHFII */
