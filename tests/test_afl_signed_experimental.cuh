#pragma once
#include "tests/test_afl.cuh"
#include "compression/afl_signed_experimental.cuh"

template <typename T, int CWARP_SIZE> class test_afl_signed: public test_afl<T, CWARP_SIZE> {
    public:
    virtual void compressData(int bit_length) {
    	if(bit_length == 13){
			std::cout<< bit_length<< "\n";
			std::cout<< this->max_size << "\n";
		}
        run_afl_compress_signed_gpu <T, CWARP_SIZE> (bit_length, this->dev_data, this->dev_out, this->max_size);
    }

    virtual void decompressData(int bit_length) {
        run_afl_decompress_signed_gpu <T, CWARP_SIZE> (bit_length, this->dev_out, this->dev_data, this->max_size);

        if(bit_length == 13){
                    	unsigned int *host_out;
                    	mmCudaMallocHost(this->manager, (void **) &host_out, this->max_size);
                    	gpuErrchk(cudaMemcpy(host_out, this->dev_out, this->max_size, cudaMemcpyDeviceToHost));
                		std::cout << "\nOutput\n";
                    	for(int i = 0 ; i < 10; i++){
                    		printf("%i:  ", host_out[i]);
                    		printBits(sizeof(int), &(host_out[i]));
                    	}
                    }
    }

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

    void printBits(size_t const size, void const * const ptr)
    {
        unsigned char *b = (unsigned char*) ptr;
        unsigned char byte;
        int i, j;

        for (i=size-1;i>=0;i--)
        {
            for (j=7;j>=0;j--)
            {
                byte = (b[i] >> j) & 1;
                printf("%u", byte);
            }
        }
        puts("");
    }
};
