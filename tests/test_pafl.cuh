#ifndef TEST_PAFL_CUH_YCIJ79CD
#define TEST_PAFL_CUH_YCIJ79CD
#include "test_base.cuh"
#include "compression/pafl_gpu.cuh"

template <typename T, char CWARP_SIZE> 
class test_pafl: public test_base<T, CWARP_SIZE> 
{
    public: 
        virtual void allocateMemory() {
            /* mmCudaMallocHost(this->manager,(void**)&this->host_data, this->data_size); */
            /* mmCudaMallocHost(this->manager,(void**)&this->host_data2, this->data_size); */

            /* // allocate maximal compressed data size rather than independent allocations for each compression ratio- improves testing time */
            /* mmCudaMalloc(this->manager, (void **) &this->dev_out, this->data_size); */ 
            /* mmCudaMalloc(this->manager, (void **) &this->dev_data, this->data_size); */



            mmCudaMalloc(this->manager, (void **) &this->dev_data_patch_values, outlier_data_size);
            mmCudaMalloc(this->manager, (void **) &this->dev_data_patch_index, (outlier_count + 1024) * sizeof(unsigned long));

            mmCudaMalloc(this->manager, (void **) &this->dev_data_patch_count, sizeof(unsigned long));

            test_base <T, CWARP_SIZE>::allocateMemory();

            /* mmCudaMalloc(this->manager, (void **) &this->dev_queue_patch_count, sizeof(unsigned long)); */
            /* mmCudaMalloc(this->manager, (void **) &this->dev_queue_patch_index, outlier_count * sizeof(unsigned long)); */
            /* mmCudaMalloc(this->manager, (void **) &this->dev_queue_patch_values, outlier_data_size); */
        }

        virtual void setup(unsigned long max_size) {
            test_base <T, CWARP_SIZE>::setup(max_size);
            this->outlier_percent = 0.1;

            this->outlier_count = max_size * this->outlier_percent;
            /* this->outlier_count = 100; */
            this->outlier_data_size = (this->outlier_count + 1024) * sizeof(T);
        }

        virtual void initializeData(int bit_length) {
            int outlier_bits=3;
            big_random_block_with_outliers(this->max_size, this->outlier_count, bit_length, outlier_bits, this->host_data);

            this->comp_h.bit_length = bit_length;
            this->comp_h.patch_bit_length = outlier_bits;
        }

        // Clean up before compression
        virtual void cleanBeforeCompress() {
            test_base <T, CWARP_SIZE>::cleanBeforeCompress();
            cudaMemset(this->dev_data_patch_count, 0,  sizeof(unsigned long)); 
            /* cudaMemset(this->dev_data_patch_index, 0,  (outlier_count + 1024) * sizeof(unsigned long)); */ 
            /* cudaMemset(this->dev_data_patch_values, 0,  (outlier_count + 1024) * sizeof(T)); */ 
            /* cudaMemset(this->dev_queue_patch_count, 0, sizeof(unsigned long)); */ 
        }

        virtual void compressData(int bit_length) {
            /* run_pafl_compress_gpu_alternate <T,CWARP_SIZE> ( */
            /*     this->comp_h, */
            /*     this->dev_data, */
            /*     this->dev_out, */
            /*     this->max_size, */

            /*     this->dev_queue_patch_values, */
            /*     this->dev_queue_patch_index, */
            /*     this->dev_queue_patch_count, */

            /*     this->dev_data_patch_values, */
            /*     this->dev_data_patch_index, */
            /*     this->dev_data_patch_count */
            /*     ); */

            run_pafl_compress_gpu_alternate <T,CWARP_SIZE> (
                this->comp_h,
                this->dev_data,
                this->dev_out,
                this->max_size,

                this->dev_data_patch_values,
                this->dev_data_patch_index,
                this->dev_data_patch_count,

                this->dev_data_patch_values,
                this->dev_data_patch_index,
                this->dev_data_patch_count
                );

    cudaErrorCheck();
        }

        virtual void decompressData(int bit_length) {
            run_pafl_decompress_gpu < T, CWARP_SIZE> (
                this->comp_h, 
                this->dev_out, 
                this->dev_data, 
                this->max_size,

                this->dev_data_patch_values,
                this->dev_data_patch_index,
                this->dev_data_patch_count
                /* this->dev_queue_patch_count */
                );
        }

    protected:
        unsigned long *dev_data_patch_index;
        unsigned long *dev_data_patch_count;
        T *dev_data_patch_values;

        /* unsigned long *dev_queue_patch_count; */
        /* unsigned long *dev_queue_patch_index; */
        /* T *dev_queue_patch_values; */ 

        unsigned long outlier_count;
        unsigned long outlier_data_size;
        float outlier_percent;
        pafl_header comp_h;
};

#endif /* end of include guard: TEST_PAFL_CUH_YCIJ79CD */
