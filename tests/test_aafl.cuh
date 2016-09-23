#ifndef TEST_AAFL_CUH_BUYDHFII
#define TEST_AAFL_CUH_BUYDHFII

#include "test_base.cuh"
#include "compression/aafl_gpu.cuh"

template <typename T, int CWARP_SIZE> 
class test_aafl: public virtual test_base<T, CWARP_SIZE> 
{
    public: 
        virtual void allocateMemory() {
            test_base<T, CWARP_SIZE>::allocateMemory();
            iner_allocateMemory();
        }

        virtual void setup(unsigned long max_size) {
            test_base<T, CWARP_SIZE>::setup(max_size);
            iner_setup(max_size);
        }

        virtual void initializeData(int bit_length) {
            big_random_block(this->max_size, bit_length, this->host_data);
        }

        // Clean up before compression
        virtual void cleanBeforeCompress() {
            test_base<T, CWARP_SIZE>::cleanBeforeCompress();
            iner_cleanBeforeCompress();
        }

        virtual void compressData(int bit_length) {
            run_aafl_compress_gpu <T, CWARP_SIZE> (this->dev_data_compressed_data_register, this->dev_data_bit_lenght, this->dev_data_position_id, this->dev_data, this->dev_out, this->max_size);
        }

        virtual void decompressData(int bit_length) {
            run_aafl_decompress_gpu <T, CWARP_SIZE> (this->dev_data_bit_lenght, this->dev_data_position_id, this->dev_out, this->dev_data, this->max_size);
        }

        virtual void print_compressed_data_size(){
            //TODO: fix this
            unsigned long tmp;
            cudaMemcpy(&tmp, this->dev_data_compressed_data_register, sizeof(unsigned long), cudaMemcpyDeviceToHost);
                printf("Comp ratio %f",  (float)this->max_size / (tmp + this->compression_blocks_count * (sizeof(long) + sizeof(char))));
                printf(" %d %lu %ld %ld\n" , this->bit_length, this->max_size, this->data_size, this->compressed_data_size);
        }

    protected:
        unsigned char *dev_data_bit_lenght;
        unsigned long *dev_data_position_id;
        unsigned long compression_blocks_count;
        unsigned long *dev_data_compressed_data_register;

        virtual void iner_allocateMemory(){
            mmCudaMalloc(this->manager, (void **) &this->dev_data_compressed_data_register, sizeof(long));
            mmCudaMalloc(this->manager, (void **) &this->dev_data_bit_lenght, compression_blocks_count * sizeof(unsigned char));
            mmCudaMalloc(this->manager, (void **) &this->dev_data_position_id, compression_blocks_count * sizeof(unsigned long));

        }

        virtual void iner_cleanBeforeCompress(){
            cudaMemset(this->dev_data_compressed_data_register, 0, sizeof(unsigned long)); 
            cudaMemset(this->dev_data_bit_lenght, 0, compression_blocks_count * sizeof(unsigned char));
            cudaMemset(this->dev_data_position_id, 0, compression_blocks_count * sizeof(unsigned long));
        }

        virtual void iner_setup(unsigned long max_size) {
            this->compression_blocks_count = (this->compressed_data_size + (sizeof(T) * CWARP_SIZE) - 1) / (sizeof(T) * CWARP_SIZE);
        }
};

#endif /* end of include guard: TEST_AAFL_CUH_BUYDHFII */
