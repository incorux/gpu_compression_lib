#ifndef DATA_CUH_8QZALZKW
#define DATA_CUH_8QZALZKW 1
#include <stdio.h>
#include <iostream>

// Tools
template <typename T> void big_random_block ( unsigned long size, int limit_bits, T *data);
template <typename T> void big_random_block_with_outliers ( unsigned long size, int outlier_count, int limit_bits, int outlier_bits,  T *data);

template <typename T> int compare_arrays ( T *in1, T *in2, unsigned long size);

template <typename T>
void big_random_block_with_decreasing_values( unsigned long size, int limit_bits, T *data);

template <typename T1, typename T2, typename T3>
void inline compare_arrays_element_print(T1 i, T2 a, T3 b)
{
    ;//std::cout<< "Error at " << i << "element "<< a << " != " << b << "\n";
}

#define DEBUG_DATA 0

#ifdef DEBUG_DATA
# define DPRINT(x) printf x
#else
# define DPRINT(x) do {} while (0)
#endif

#endif /* end of include guard: DATA_CUH_8QZALZKW */
