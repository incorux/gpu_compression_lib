#ifndef TEST_DELTA_CUH_ISZ6QCRW
#define TEST_DELTA_CUH_ISZ6QCRW
#include "test_base.cuh"
#include "compression/delta_gpu.cuh"

template <typename T, int CWARP_SIZE> 
class test_delta: public test_base<T, CWARP_SIZE> {
    public: 
        virtual void setup(unsigned long max_size) {
            test_base <T,CWARP_SIZE>::setup(max_size);
            this->spoints_count = max_size;
            this->cword = 2;
        }

        virtual void allocateMemory() {
            test_base <T,CWARP_SIZE>::allocateMemory();
            mmCudaMalloc(this->manager, (void **) &this->dev_spoints, this->spoints_count);
        }

        virtual void compressData(int bit_length) {
            printf("!!!Hallo\n");

            delta_compress_gpu <int> <<<1,32>>>(this->dev_data, this->dev_out, this->dev_spoints, (unsigned int)bit_length, this->data_size, this->spoints_count);
            // TODO: in DELTA we ignore bit_length - maybe rewrite run method
        }

        virtual void decompressData(int bit_length) {
            printf("!!!Hallo\n");
            // TODO: in DELTA we ignore bit_length - maybe rewrite run method
        }

    protected:
        T *dev_spoints;
        unsigned long spoints_count;
};

#endif /* end of include guard: TEST_DELTA_CUH_ISZ6QCRW */
