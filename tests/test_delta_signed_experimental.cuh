#pragma once
#include "test_delta.cuh"
#include "compression/delta_signed_experimental.cuh"
template <typename T, char CWARP_SIZE>
class test_delta_signed: public virtual test_delta<T, CWARP_SIZE>
{
    public:

        virtual void initializeData(int bit_length) {
            big_random_block(this->max_size, this->bit_length-1, this->host_data);
            // On signed types this will make all odd values negative
            if (std::numeric_limits<T>::is_signed)
                for (unsigned long i = 0; i < this->max_size; i++)
                    if (i%2)
                        this->host_data[i] *= -1;
            if(bit_length == 13){
                		for(int i = 0 ; i < 100; i++){
            				printf("%i ", this->host_data[i]);
            			}
            }
        }

        virtual void compressData(int bit_length) {
            run_delta_afl_compress_signed_gpu <T, CWARP_SIZE> (bit_length, this->dev_data, this->dev_out, this->dev_data_block_start, this->max_size);
        }

        virtual void decompressData(int bit_length) {
            run_delta_afl_decompress_signed_gpu <T, CWARP_SIZE> (bit_length, this->dev_out, this->dev_data_block_start, this->dev_data, this->max_size);

            if(bit_length == 13){
            	signed int * dev_data_block_start_host;
            	mmCudaMallocHost(this->manager, (void **) &dev_data_block_start_host, this->compression_blocks_count * sizeof(unsigned long));
            	gpuErrchk(cudaMemcpy(dev_data_block_start_host, this->dev_data_block_start, this->compression_blocks_count * sizeof(unsigned long), cudaMemcpyDeviceToHost));
        		std::cout << "\nHelper\n";
            	for(int i = 0 ; i < (10< this->compression_blocks_count ? 10 : this->compression_blocks_count)  ; i++){
            		printf("%i:  ", dev_data_block_start_host[i]);
            	}
            }
            if(bit_length == 13){
                        	int * host_out;
                        	mmCudaMallocHost(this->manager, (void **) &host_out, this->compressed_data_size);
                        	gpuErrchk(cudaMemcpy(host_out, this->dev_out, this->compressed_data_size, cudaMemcpyDeviceToHost));
                    		std::cout << "\nOutput\n";
                        	for(int i = 0 ; i < 10; i++){
                        		printf("%i:  ", host_out[i]);
                        	}
                        }
        }
};
