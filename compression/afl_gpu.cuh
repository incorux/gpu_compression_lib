#ifndef afl_H
#define afl_H 1

template < typename T, char CWORD_SIZE, char CWARP_SIZE > 
__host__ void run_afl_decompress_gpu    ( int bit_length  , T *data            , T *compressed_data   , unsigned long length);

template < typename T, char CWORD_SIZE, char CWARP_SIZE >
__host__ void run_afl_compress_gpu      ( int bit_length  , T *compressed_data , T *decompressed_data , unsigned long length);

template < typename T, char CWORD_SIZE, char CWARP_SIZE >
__host__ void run_afl_decompress_value_gpu(int bit_length , T *compressed_data , T *data              , unsigned long length);

template < typename T, char CWORD_SIZE, char CWARP_SIZE >
__global__ void afl_decompress_value_gpu (int bit_length , T *compressed_data , T * decompress_data , unsigned long length);
template < typename T, char CWORD_SIZE, char CWARP_SIZE >
__global__ void afl_compress_gpu        ( int bit_length , T *data            , T *compressed_data  , unsigned long length);
template < typename T, char CWORD_SIZE, char CWARP_SIZE >
__global__ void afl_decompress_gpu      ( int bit_length , T *compressed_data , T * decompress_data , unsigned long length);

template < typename T, char CWORD_SIZE, char CWARP_SIZE >
__device__ __host__ T afl_decompress_base_value_gpu ( int bit_length, T *compressed_data, unsigned long pos);
template < typename T, char CWORD_SIZE, char CWARP_SIZE >
__device__ __host__ void afl_compress_base_gpu( int bit_length, unsigned long data_id, unsigned long comp_data_id, T *data, T *compressed_data, unsigned long length);

template < typename T, char CWORD_SIZE, char CWARP_SIZE >
__device__ __host__ void afl_decompress_base_gpu( int bit_length, unsigned long comp_data_id, unsigned long data_id, T *compressed_data, T *data,unsigned long length);

#endif
