#ifndef TEST_PAFL_CUH_YCIJ79CD
#define TEST_PAFL_CUH_YCIJ79CD
#include "test_base.cuh"
#include "compression/pafl_gpu.cuh"

template <typename T, int CWARP_SIZE> 
class test_pafl: public test_base<T, CWARP_SIZE> 
{
    public: 
        virtual void allocateMemory() {
            mmCudaMallocHost(this->manager,(void**)&this->host_data, this->data_size);
            mmCudaMallocHost(this->manager,(void**)&this->host_data2, this->data_size);

            // allocate maximal compressed data size rather than independent allocations for each compression ratio- improves testing time
            mmCudaMalloc(this->manager, (void **) &this->dev_out, this->data_size); 
            mmCudaMalloc(this->manager, (void **) &this->dev_data, this->data_size);

            mmCudaMalloc(this->manager, (void **) &this->dev_data_patch_count, sizeof(int));
            mmCudaMalloc(this->manager, (void **) &this->dev_data_patch_index, outlier_data_size);
            mmCudaMalloc(this->manager, (void **) &this->dev_data_patch_values, outlier_data_size);
            mmCudaMalloc(this->manager, (void **) &this->dev_queue_patch_count, sizeof(int));
            mmCudaMalloc(this->manager, (void **) &this->dev_queue_patch_index, outlier_data_size);
            mmCudaMalloc(this->manager, (void **) &this->dev_queue_patch_values, outlier_data_size);
        }

        virtual void setup(unsigned long max_size) {
            this->max_size = max_size;
            this->cword = sizeof(T) * 8;
            this->data_size = max_size * sizeof(T);

            this->outlier_count = max_size * this->outlier_percent;
            this->outlier_data_size = this->outlier_count * sizeof(T);

            // for size less then cword we actually will need more space than original data
            this->compressed_data_size = (max_size < this->cword  ? this->cword : max_size) * sizeof(T);
        }

        virtual void initializeData(int bit_length) {
            int outlier_bits=3;
            big_random_block_with_outliers(this->max_size, this->outlier_count, bit_length, outlier_bits, this->host_data);

            this->comp_h.bit_length = bit_length;
            this->comp_h.patch_bit_length = outlier_bits;
        }

        // Clean up before compression
        virtual void cleanBeforeCompress() {
            cudaMemset(this->dev_out, 0, this->data_size); 
            cudaMemset(this->dev_data_patch_count, 0, sizeof(int)); 
            cudaMemset(this->dev_queue_patch_count, 0, sizeof(int)); 
        }

        virtual void compressData(int bit_length) {
            run_pafl_compress_gpu_alternate(
                this->comp_h,
                this->dev_data,
                this->dev_out,
                this->max_size,

                this->dev_queue_patch_values,
                this->dev_queue_patch_index,
                this->dev_queue_patch_count,

                this->dev_data_patch_values,
                this->dev_data_patch_index,
                this->dev_data_patch_count
                );
        }

        virtual void decompressData(int bit_length) {
            run_pafl_decompress_gpu(
                this->comp_h, 
                this->dev_out, 
                this->dev_data, 
                this->max_size,

                this->dev_data_patch_values,
                this->dev_data_patch_index,
                this->dev_queue_patch_count
                );
        }

        test_pafl(float outlier_percent): outlier_percent(outlier_percent) {};

    protected:
        T *dev_data_patch_index, *dev_data_patch_values, *dev_data_patch_count;
        T *dev_queue_patch_index, *dev_queue_patch_values, *dev_queue_patch_count;
        int outlier_count;
        int outlier_data_size;
        float outlier_percent;
        pafl_header comp_h;
};

#endif /* end of include guard: TEST_PAFL_CUH_YCIJ79CD */
