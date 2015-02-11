#include "data.cuh"
#include "../compression/macros.cuh"

static unsigned long __xorshf96_x=123456789, __xorshf96_y=362436069, __xorshf96_z=521288629;

unsigned long xorshf96(void) {          //period 2^96-1
    unsigned long t;
        __xorshf96_x ^= __xorshf96_x << 16;
        __xorshf96_x ^= __xorshf96_x >> 5;
        __xorshf96_x ^= __xorshf96_x << 1;

        t = __xorshf96_x;
        __xorshf96_x = __xorshf96_y;
        __xorshf96_y = __xorshf96_z;
        __xorshf96_z = t ^ __xorshf96_x ^ __xorshf96_y;

        return __xorshf96_z;
}
// This is only for test purposes so it is optimized for speed (true randomness is not needed)

template <typename T>
void big_random_block( unsigned long size, int limit_bits, T *data) 
{
    T mask = NBITSTOMASK(limit_bits);
    for (unsigned long i = 0; i < size; i++)
        data[i] = xorshf96() & mask;
}

template <typename T>
void big_random_block_with_outliers( unsigned long size, int outlier_count, int limit_bits, int outlier_bits,  T *data) 
{
    big_random_block(size, limit_bits, data);
    unsigned int mask = NBITSTOMASK(limit_bits + outlier_bits);

    for (int i = 0; i < outlier_count; ++i) {
        int p = xorshf96() % size;
        data[ p ] = xorshf96() & mask;
    }
}

void compare_arrays_element_print(long i, long a, long b)
{
    DPRINT(("Error at %ld element (%ld != %ld)\n ", i, a, b));
}

void compare_arrays_element_print(long i, int a, int b)
{
    DPRINT(("Error at %ld element (%d != %d)\n ", i, a, b));
}
template <typename T>
int compare_arrays(T *in1, T *in2, unsigned long size)
{
    unsigned long count_errors = 0;
    for(unsigned long i = 0; i < size; i++) {
        if(in1[i] != in2[i]) {
            count_errors += 1;
            /*compare_arrays_element_print(i, in1[i], in2[i]);*/
        }
    }
    if (count_errors)
        DPRINT(("<================== ERROR ============= size = %ld errors = %ld\n", size, count_errors));
    return count_errors;
}

template void big_random_block <int> ( unsigned long size, int limit_bits, int *data);
template void big_random_block_with_outliers <int> ( unsigned long size, int outlier_count, int limit_bits, int outlier_bits,  int *data);

template void big_random_block <long> ( unsigned long size, int limit_bits, long *data);
template void big_random_block_with_outliers <long> ( unsigned long size, int outlier_count, int limit_bits, int outlier_bits,  long *data);

template int compare_arrays <long> (long *in1, long *in2, unsigned long size);
template int compare_arrays <int> (int *in1, int *in2, unsigned long size);
