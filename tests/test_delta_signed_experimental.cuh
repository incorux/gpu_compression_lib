#pragma once
#include "test_delta.cuh"
#include "compression/delta_signed_experimental.cuh"
template <typename T, char CWARP_SIZE>
class test_delta_signed: public virtual test_delta<T, CWARP_SIZE>
{
    public:

        virtual void initializeData(int bit_length) {
            big_random_block_with_decreasing_values ((unsigned long)this->max_size, bit_length, this->host_data);

            // On signed types this will make all odd values negative
            if (std::numeric_limits<T>::is_signed)
                for (unsigned long i = 0; i < this->max_size; i++)
                    if (i%2) this->host_data[i] *= -1;
        }

        virtual void compressData(int bit_length) {
            run_delta_afl_compress_signed_gpu <T, CWARP_SIZE> (bit_length, this->dev_data, this->dev_out, this->dev_data_block_start, this->max_size);
        }

        virtual void decompressData(int bit_length) {
            run_delta_afl_decompress_signed_gpu <T, CWARP_SIZE> (bit_length, this->dev_out, this->dev_data_block_start, this->dev_data, this->max_size);
        }
};
