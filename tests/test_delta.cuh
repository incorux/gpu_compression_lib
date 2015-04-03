#ifndef TEST_DELTA_CUH_ISZ6QCRW
#define TEST_DELTA_CUH_ISZ6QCRW
#include "test_afl.cuh"
#include "compression/delta_gpu.cuh"

template <typename T, int CWARP_SIZE> class test_delta: public test_afl<T, CWARP_SIZE> {

    public: 
        virtual void setup(int max_size) {
            test_afl::setup(max_size);
            this->spoints_count = max_size;
            
        }

        virtual void allocateMemory() {
            mmCudaMallocHost(this->manager,(void**)&this->host_data, this->data_size);
            mmCudaMallocHost(this->manager,(void**)&this->host_data2, this->data_size);

            // allocate maximal compressed data size - improves testing time
            mmCudaMalloc(this->manager, (void **) &this->dev_out, this->data_size); 
            mmCudaMalloc(this->manager, (void **) &this->dev_data, this->data_size);

            mmCudaMalloc(this->manager, (void **) &this->dev_spoints, spoints_count);
        }

        virtual void compressData(int bit_length) {
            // TODO: in DELTA we ignore bit_length - maybe rewrite run method
        }

        virtual void decompressData(int bit_length) {
            // TODO: in DELTA we ignore bit_length - maybe rewrite run method
        }

    protected:
        T *dev_spoints;
        int spoints_count;
};

#endif /* end of include guard: TEST_DELTA_CUH_ISZ6QCRW */
